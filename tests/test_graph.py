"""Tests for graph construction and routing.

These cover the wiring that a LangGraph major-version upgrade is most likely to
break: node/edge topology, the ``Send``-based fan-out to parallel retrievers,
and the reviewer loop. None of them call an LLM.
"""

import pytest

from honegumi_rag_assistant.app_config import settings
from honegumi_rag_assistant.orchestrator import (
    build_graph,
    continue_to_retrieval,
    route_after_review,
)

BASE_NODES = {
    "__start__",
    "__end__",
    "select_parameters",
    "generate_skeleton",
    "plan_retrieval",
    "retrieve_parallel",
    "write_code",
}


class TestBuildGraph:
    def test_compiles_without_reviewer(self):
        graph = build_graph()
        assert set(graph.get_graph().nodes) == BASE_NODES

    def test_compiles_with_reviewer(self):
        graph = build_graph(enable_review=True)
        assert set(graph.get_graph().nodes) == BASE_NODES | {"review_code"}

    def test_reviewer_off_by_default(self):
        assert "review_code" not in set(build_graph().get_graph().nodes)

    def test_deprecated_skip_review_still_suppresses_reviewer(self):
        """skip_review is deprecated but documented as still honoured."""
        graph = build_graph(skip_review=True, enable_review=True)
        assert "review_code" not in set(graph.get_graph().nodes)

    def test_graph_is_checkpointed(self):
        """Runs are isolated by thread_id, which requires a checkpointer."""
        assert build_graph().checkpointer is not None


class TestContinueToRetrieval:
    """Fan-out routing. Each returned Send is one parallel branch."""

    def test_no_queries_skips_to_code_writer(self):
        sends = continue_to_retrieval({"retrieval_queries": []})
        assert len(sends) == 1
        assert sends[0].node == "write_code"

    def test_missing_key_skips_to_code_writer(self):
        sends = continue_to_retrieval({})
        assert len(sends) == 1
        assert sends[0].node == "write_code"

    def test_no_vectorstore_skips_to_code_writer(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_vectorstore_path", "")
        sends = continue_to_retrieval({"retrieval_queries": ["q1", "q2"]})
        assert len(sends) == 1
        assert sends[0].node == "write_code"

    def test_nonexistent_vectorstore_path_skips(self, monkeypatch):
        monkeypatch.setattr(
            settings, "retrieval_vectorstore_path", "/nope/not/a/real/store"
        )
        sends = continue_to_retrieval({"retrieval_queries": ["q1"]})
        assert len(sends) == 1
        assert sends[0].node == "write_code"

    @pytest.mark.parametrize("n_queries", [1, 3, 7])
    def test_fans_out_one_send_per_query(self, monkeypatch, tmp_path, n_queries):
        monkeypatch.setattr(settings, "retrieval_vectorstore_path", str(tmp_path))
        queries = [f"query {i}" for i in range(n_queries)]

        sends = continue_to_retrieval({"retrieval_queries": queries})

        assert len(sends) == n_queries
        assert all(s.node == "retrieve_parallel" for s in sends)
        # Each branch must carry its own query and a distinct index, or the
        # retrievers would duplicate work and the debug output would mislabel.
        assert [s.arg["query"] for s in sends] == queries
        assert [s.arg["index"] for s in sends] == list(range(n_queries))

    def test_file_path_is_not_treated_as_a_store(self, monkeypatch, tmp_path):
        """The store must be a directory; a file of the same name is not one."""
        bogus = tmp_path / "store"
        bogus.write_text("not a directory")
        monkeypatch.setattr(settings, "retrieval_vectorstore_path", str(bogus))
        sends = continue_to_retrieval({"retrieval_queries": ["q1"]})
        assert sends[0].node == "write_code"


class TestRouteAfterReview:
    def test_approved_code_ends_the_run(self):
        assert route_after_review({"final_code": "print(1)"}) == "__end__"

    def test_missing_final_code_returns_to_writer(self):
        assert route_after_review({"candidate_code": "print(1)"}) == "write_code"

    def test_empty_final_code_returns_to_writer(self):
        assert route_after_review({"final_code": ""}) == "write_code"
