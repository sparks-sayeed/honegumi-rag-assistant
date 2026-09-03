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
    def test_warms_the_store_before_serving(self, monkeypatch):
        """Warming after ``run`` would be dead code: ``run`` blocks until the
        server is shut down, so the first caller would pay the 1.2 GB load that
        warming exists to absorb.
        """
        order = []
        monkeypatch.setattr(mcp_server, "warm_vectorstore", lambda: order.append("warm"))
        monkeypatch.setattr(mcp_server.mcp, "run", lambda **kw: order.append(("run", kw)))

        mcp_server.main()

        assert order[0] == "warm"
        assert order[1][0] == "run"

    def test_serves_over_http(self, monkeypatch):
        """stdio would make the server unreachable once it is deployed: the host
        would have to launch it as a local subprocess."""
        captured = {}
        monkeypatch.setattr(mcp_server, "warm_vectorstore", lambda: None)
        monkeypatch.setattr(mcp_server.mcp, "run", lambda **kw: captured.update(kw))

        mcp_server.main()

        assert captured["transport"] == "streamable-http"
