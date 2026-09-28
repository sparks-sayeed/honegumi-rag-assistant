"""Tests for the MCP server that exposes the pipeline to other agents.

The server itself is thin -- one decorated function delegating to
``run_from_text``. What is *not* thin is the tool description: it is the prompt
a calling model reads when deciding whether to invoke this tool and what to pass
it. In manual testing it is what made Claude refuse a vague request ("I want to
optimize my experiment") and instead ask for objectives and ranges, then
restructure the answer into a complete problem statement.

That behaviour rests entirely on text inside a docstring, where nothing marks it
as load-bearing. Someone tidying up "just a docstring" would break the tool's
judgement without breaking a single line of code, and the failure would be
silent -- generation still works, it just stops asking first. These tests pin
the parts of that contract that matter.
"""

import asyncio

import pytest

from honegumi_rag_assistant import mcp_server


@pytest.fixture(scope="module")
def tool():
    """The single registered tool, as a calling client would receive it."""
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert len(tools) == 1, f"expected exactly one tool, got {[t.name for t in tools]}"
    return tools[0]


class TestModuleImport:
    """Importing the module must not start a server."""

    def test_import_is_side_effect_free(self):
        """If ``mcp.run()`` ran at import, this module would never finish loading.

        The real assertion is that collection got this far at all -- a missing
        ``if __name__ == "__main__"`` guard would hang the whole suite rather
        than fail it. Serving is reachable only through ``main``.
        """
        assert callable(mcp_server.main)


class TestToolRegistration:
    def test_tool_name(self, tool):
        """The name is part of the public contract; hosts namespace tools by it."""
        assert tool.name == "generate_bo_code"

    def test_takes_a_single_required_problem_string(self, tool):
        schema = tool.input_schema
        assert schema["properties"]["problem"]["type"] == "string"
        assert schema["required"] == ["problem"]


class TestToolDescription:
    """The description is a prompt. These pin what it has to keep saying."""

    def test_states_which_inputs_are_required(self, tool):
        """Without objectives and ranges the pipeline invents both, and the
        generated script is confidently wrong rather than obviously broken."""
        description = tool.description.lower()
        assert "required" in description
        assert "maximise" in description or "maximize" in description
        assert "ranges" in description or "bounds" in description

    def test_tells_the_caller_to_ask_before_guessing(self, tool):
        """The instruction that makes a calling model gather missing details
        instead of calling with a vague string."""
        description = tool.description.lower()
        assert "ask" in description

    def test_describes_the_problem_shape_not_just_the_jargon(self, tool):
        """Users say "get the best yield", never "Bayesian optimization". The
        description carries concrete examples so a model can match on the shape
        of the request rather than on a phrase nobody types."""
        description = tool.description.lower()
        assert "yield" in description
        assert any(word in description for word in ("minimise", "minimize"))

    def test_says_what_comes_back(self, tool):
        """A caller needs to know it receives a script, not a report or a plan."""
        description = tool.description.lower()
        assert "script" in description


class TestMain:
    """``main`` decides whether serving is safe before it serves."""

    @pytest.fixture
    def served(self, monkeypatch):
        """Capture what ``main`` would have served, without serving it."""
        captured = {}
        monkeypatch.setattr(mcp_server, "warm_vectorstore",
                            lambda: captured.setdefault("order", []).append("warm"))
        monkeypatch.setattr(mcp_server.mcp, "streamable_http_app",
                            lambda **kw: "INNER_APP")

        def fake_serve(app, **kw):
            captured.setdefault("order", []).append("serve")
            captured["app"] = app
            captured.update(kw)

        monkeypatch.setattr(mcp_server.uvicorn, "run", fake_serve)
        return captured

    def test_warms_the_store_before_serving(self, served):
        """Warming after serving would be dead code: uvicorn blocks until
        shutdown, so the first caller would pay the 1.2 GB load that warming
        exists to absorb.
        """
        mcp_server.main()
        assert served["order"] == ["warm", "serve"]

    def test_binds_to_the_configured_host_and_port(self, served, monkeypatch):
        """The container sets MCP_HOST=0.0.0.0. Hardcoding loopback would make
        ``docker run -p`` publish a port that refuses every connection.
        """
        monkeypatch.setattr(mcp_server.settings, "mcp_host", "0.0.0.0")
        monkeypatch.setattr(mcp_server.settings, "mcp_port", 9999)
        monkeypatch.setattr(mcp_server.settings, "mcp_auth_token", "secret")

        mcp_server.main()

        assert served["host"] == "0.0.0.0"
        assert served["port"] == 9999

    def test_default_host_is_loopback(self):
        """Safe by default: a server started on a laptop stays on that laptop."""
        from honegumi_rag_assistant.app_config import Settings

        assert Settings().mcp_host == "127.0.0.1"


class TestRefusesToServeUnprotected:
    """The failure mode this guard exists for is financial, not technical."""

    def test_non_loopback_without_a_token_will_not_start(self, monkeypatch):
        monkeypatch.setattr(mcp_server.settings, "mcp_host", "0.0.0.0")
        monkeypatch.setattr(mcp_server.settings, "mcp_auth_token", "")

        with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
            mcp_server.main()

    def test_it_refuses_before_doing_any_work(self, monkeypatch):
        """The check must precede warmup -- otherwise a misconfigured server
        spends 30 seconds loading a model it is about to refuse to serve."""
        warmed = []
        monkeypatch.setattr(mcp_server, "warm_vectorstore", lambda: warmed.append(1))
        monkeypatch.setattr(mcp_server.settings, "mcp_host", "0.0.0.0")
        monkeypatch.setattr(mcp_server.settings, "mcp_auth_token", "")

        with pytest.raises(RuntimeError):
            mcp_server.main()

        assert warmed == []

    def test_loopback_without_a_token_is_allowed(self, monkeypatch):
        """Local development must not need a token."""
        monkeypatch.setattr(mcp_server, "warm_vectorstore", lambda: None)
        monkeypatch.setattr(mcp_server.mcp, "streamable_http_app", lambda **kw: "APP")
        monkeypatch.setattr(mcp_server.uvicorn, "run", lambda app, **kw: None)
        monkeypatch.setattr(mcp_server.settings, "mcp_host", "127.0.0.1")
        monkeypatch.setattr(mcp_server.settings, "mcp_auth_token", "")

        mcp_server.main()  # must not raise


class TestBearerTokenMiddleware:
    """The gate itself."""

    @staticmethod
    def _request(token_header: bytes | None):
        """Drive the middleware through one HTTP request, return (reached, sent)."""
        reached = []

        async def inner_app(scope, receive, send):
            reached.append(scope)

        middleware = mcp_server.BearerTokenMiddleware(inner_app, "s3cret")
        headers = [(b"authorization", token_header)] if token_header else []
        sent = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            sent.append(message)

        asyncio.run(middleware({"type": "http", "headers": headers}, receive, send))
        return reached, sent

    def test_correct_token_reaches_the_app(self):
        reached, sent = self._request(b"Bearer s3cret")
        assert len(reached) == 1
        assert sent == []

    def test_missing_header_is_rejected(self):
        reached, sent = self._request(None)
        assert reached == []
        assert sent[0]["status"] == 401

    def test_wrong_token_is_rejected(self):
        reached, sent = self._request(b"Bearer wrong")
        assert reached == []
        assert sent[0]["status"] == 401

    def test_bare_token_without_the_scheme_is_rejected(self):
        """``Authorization: s3cret`` is not the same as ``Bearer s3cret``."""
        reached, sent = self._request(b"s3cret")
        assert reached == []
        assert sent[0]["status"] == 401

    def test_rejection_advertises_the_scheme(self):
        """A 401 should tell the client how to authenticate."""
        _, sent = self._request(None)
        headers = dict(sent[0]["headers"])
        assert headers[b"www-authenticate"] == b"Bearer"

    def test_non_http_scopes_pass_straight_through(self):
        """Lifespan startup/shutdown events carry no headers; blocking them
        would stop the server booting at all."""
        reached = []

        async def inner_app(scope, receive, send):
            reached.append(scope)

        middleware = mcp_server.BearerTokenMiddleware(inner_app, "s3cret")

        async def noop(*args, **kwargs):
            return {}

        asyncio.run(middleware({"type": "lifespan"}, noop, noop))
        assert len(reached) == 1
