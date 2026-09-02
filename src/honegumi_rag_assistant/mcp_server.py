"""MCP server exposing the Honegumi RAG Assistant as a callable tool."""

from mcp.server.mcpserver import MCPServer

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


def main() -> None:
    """Warm the vector store, then serve until interrupted.

    Warming first means the embedding model and FAISS index are resident before
    the first caller arrives, rather than that caller waiting on them.
    """
    warm_vectorstore()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
