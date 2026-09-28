"""MCP server exposing the Honegumi RAG Assistant as a callable tool."""

import secrets
from typing import Any, Awaitable, Callable, MutableMapping

import uvicorn
from mcp.server.mcpserver import MCPServer

from .app_config import settings
from .nodes.retriever import warm_vectorstore
from .orchestrator import run_from_text

mcp = MCPServer(
    "honegumi-rag-assistant",
    title="Honegumi RAG Assistant",
    description="Generates runnable Bayesian optimization scripts from "
                "natural-language problem descriptions.",
)


@mcp.tool()
def generate_bo_code(problem: str) -> str:
    """Generate a runnable Python script that sets up a Bayesian optimization campaign.

    Use this when someone wants to find the best settings for an experiment or
    process by testing candidates iteratively -- choosing temperature and pressure
    to maximise reaction yield, tuning alloy composition for strength and cost,
    picking hyperparameters to minimise validation loss. Users rarely say
    "Bayesian optimization"; look for a goal to maximise or minimise, plus
    adjustable inputs with ranges.

    Returns a complete, executable Python script built on Ax and Honegumi, with
    parameters, objectives and constraints filled in for the specific problem,
    plus an evaluation-function stub for the user to complete.

    The `problem` argument must state, in plain English:
      - what to maximise or minimise (at least one objective) -- REQUIRED
      - the adjustable parameters and their ranges or allowed values -- REQUIRED
      - any constraints, e.g. components must sum to 100%  (optional)
      - experiment budget, batch size, or existing data     (optional)

    If the user has not given both the objectives and the parameter ranges, ask
    them before calling this tool. Output quality depends entirely on those.

    Not for: analysing data the user already has, general ML questions, or
    optimization approaches other than Bayesian optimization with Ax.
    Takes 1-3 minutes to run.
    """
    return run_from_text(problem)


# Interfaces reachable only from the machine itself.  Anything else is exposed
# to at least the local network, and needs a token.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class BearerTokenMiddleware:
    """Reject any HTTP request that does not present the shared secret.

    Implemented as plain ASGI middleware rather than through the SDK's auth
    settings, which model an OAuth resource server and require an issuer URL.
    There is no issuer here -- just one secret shared between this server and
    the people allowed to call it -- so the smaller mechanism is the honest one.

    Parameters
    ----------
    app
        The ASGI application to wrap.
    token : str
        The secret callers must present as ``Authorization: Bearer <token>``.
    """

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self._expected = f"Bearer {token}"

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":  # lifespan events carry no headers
            await self.app(scope, receive, send)
            return

        presented = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                presented = value.decode("latin-1")
                break

        # Constant-time comparison: a plain ``==`` returns as soon as two bytes
        # differ, so response timing would leak the token one character at a
        # time to anyone patient enough to measure it.
        if not secrets.compare_digest(presented, self._expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"text/plain"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return

        await self.app(scope, receive, send)


def main() -> None:
    """Warm the vector store, then serve until interrupted.

    Warming first means the embedding model and FAISS index are resident before
    the first caller arrives, rather than that caller waiting on them.

    Raises
    ------
    RuntimeError
        If the server would listen on a non-loopback interface with no
        ``MCP_AUTH_TOKEN`` set.  Failing to start is the point: an unauthenticated
        public endpoint lets anyone who finds the URL spend the Anthropic budget
        belonging to whoever deployed it.  Making that configuration impossible
        beats documenting that it is unwise.
    """
    host = settings.mcp_host
    token = settings.mcp_auth_token

    if host not in _LOOPBACK_HOSTS and not token:
        raise RuntimeError(
            f"Refusing to serve on {host} without MCP_AUTH_TOKEN. Every call to "
            "this server spends your Anthropic credit, so a reachable endpoint "
            "must require a token. Set MCP_AUTH_TOKEN, or bind to 127.0.0.1 for "
            "local use."
        )

    warm_vectorstore()

    app: Any = mcp.streamable_http_app(host=host)
    if token:
        app = BearerTokenMiddleware(app, token)
    else:
        print("MCP_AUTH_TOKEN not set - serving unauthenticated on loopback only.")

    uvicorn.run(
        app,
        host=host,
        port=settings.mcp_port,
        log_level=mcp.settings.log_level.lower(),
    )



if __name__ == "__main__":
    main()
