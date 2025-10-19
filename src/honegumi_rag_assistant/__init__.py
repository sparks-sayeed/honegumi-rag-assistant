"""
The ``honegumi_rag_assistant`` package exposes a simple API for building and
executing an agentic workflow that generates Bayesian optimisation code from
a natural language problem description.  This package uses a LangGraph
state machine under the hood to orchestrate a collection of autonomous nodes
that each perform a single function in the overall pipeline:

* The **ParameterSelector** node parses the free‑form problem text and
  selects an appropriate set of optimisation parameters via a large
  language model using the OpenAI function calling interface.
* The **SkeletonGenerator** node calls into the Honegumi template engine to
  produce a deterministic code skeleton for the Ax platform, based on the
  selected parameters.
* The **RetrieverAgent** queries a vector store of the Ax documentation to
  fetch relevant context snippets that can be passed to the code writing
  model.  If no vector store is configured the retriever gracefully
  returns an empty list.
* The **CodeWriterAgent** writes the final Python script by editing the
  skeleton in the context of the user's problem and any retrieved
  documentation.  A simple self‑check is performed to ensure the
  skeleton's structure is preserved.
* The **ReviewerAgent** performs a final sanity check on the generated
  code and either approves it or flags an error.

The root of this package exposes two helper functions:

``build_graph()``
    Construct the LangGraph state machine for the workflow.  You can
    customise or extend this graph by importing and wiring additional
    nodes.

``run()``
    Execute the pipeline on a given problem statement and write the
    generated code to disk.  This function is suitable for use in
    command‑line interfaces or notebooks.

See the module docstrings and individual classes for more details.
"""

from .orchestrator import build_graph, run  # noqa: F401
