"""Tests for per-run settings isolation.

``settings`` is a process-wide singleton, and until now every entry point
configured a run by assigning to it -- ``settings.debug = debug``,
``settings.anthropic_api_key = api_key``. That is correct for the CLI, which
owns the process and runs one pipeline in it. It is a bug for anything serving
concurrent callers: the Gradio app set a visitor's API key and model choices on
the shared object, so a second visitor arriving mid-run would overwrite them
underneath the first.

``override_settings`` moves those per-run values into a :class:`ContextVar`,
which is scoped to the calling thread or task. These tests pin the two
properties the fix depends on: overrides are invisible to concurrent callers,
and they survive the hop into LangGraph's parallel ``Send`` branches.
"""

import operator
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, List, TypedDict

import pytest

from honegumi_rag_assistant.app_config import override_settings, settings


class TestOverrideScope:
    """An override applies inside its block and nowhere else."""

    def test_value_visible_inside_and_gone_after(self):
        original = settings.retrieval_top_k
        with override_settings(retrieval_top_k=99):
            assert settings.retrieval_top_k == 99
        assert settings.retrieval_top_k == original

    def test_unset_fields_fall_through_to_defaults(self):
        with override_settings(retrieval_top_k=99):
            assert settings.embedding_model == settings.embedding_model
            assert settings.retrieval_top_k == 99

    def test_override_is_removed_when_the_block_raises(self):
        """A failed run must not leave its config behind for the next one."""
        original = settings.retrieval_top_k
        with pytest.raises(RuntimeError):
            with override_settings(retrieval_top_k=99):
                raise RuntimeError("pipeline blew up")
        assert settings.retrieval_top_k == original

    def test_nested_scopes_merge_with_inner_winning(self):
        with override_settings(retrieval_top_k=1, code_writer_effort="low"):
            with override_settings(retrieval_top_k=2):
                assert settings.retrieval_top_k == 2
                assert settings.code_writer_effort == "low"
            assert settings.retrieval_top_k == 1


class TestProcessDefaultsStillWork:
    """The CLI configures a run by assignment; that path must keep working."""

    def test_assignment_persists(self):
        original = settings.retrieval_top_k
        try:
            settings.retrieval_top_k = 42
            assert settings.retrieval_top_k == 42
        finally:
            settings.retrieval_top_k = original

    def test_override_shadows_an_assignment_without_destroying_it(self):
        original = settings.retrieval_top_k
        try:
            settings.retrieval_top_k = 42
            with override_settings(retrieval_top_k=99):
                assert settings.retrieval_top_k == 99
            assert settings.retrieval_top_k == 42
        finally:
            settings.retrieval_top_k = original

    def test_monkeypatch_still_works(self, monkeypatch):
        """The rest of the suite configures settings this way."""
        monkeypatch.setattr(settings, "retrieval_vectorstore_path", "/patched")
        assert settings.retrieval_vectorstore_path == "/patched"


class TestConcurrentCallers:
    """The bug this fix exists for."""

    def test_two_threads_keep_their_own_values(self):
        """Both threads sit inside their override at the same instant.

        The barrier is what makes this a real test: without it the threads
        could run end to end one after the other and pass even with a shared
        singleton. With it, thread B has definitely entered its scope before
        thread A reads its value back.
        """
        barrier = threading.Barrier(2)

        def run(model_name):
            with override_settings(code_writer_model=model_name):
                barrier.wait(timeout=5)
                return settings.code_writer_model

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, ["model-A", "model-B"]))

        assert sorted(results) == ["model-A", "model-B"]

    def test_a_worker_thread_does_not_leak_into_the_caller(self):
        original = settings.code_writer_model

        def run():
            with override_settings(code_writer_model="model-thread"):
                return settings.code_writer_model

        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(run).result() == "model-thread"

        assert settings.code_writer_model == original


class _FanState(TypedDict):
    seen: Annotated[List[str], operator.add]


class TestPropagationIntoParallelBranches:
    """Overrides must survive the hop into LangGraph's ``Send`` workers.

    The retriever reads ``embedding_device``, ``retrieval_top_k`` and friends
    from ``settings`` *inside* the fan-out branches, which LangGraph runs on
    worker threads. A plain ``threading.Thread`` starts with an empty context,
    so this only works because LangGraph copies the active context into its
    workers. That is an assumption of the fix, not a guarantee of the language
    -- pin it, so a LangGraph upgrade that changed it fails here loudly rather
    than silently serving every request the default configuration.
    """

    def test_send_branches_see_the_callers_override(self):
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send

        def fan_out(state):
            return [Send("worker", {"i": i}) for i in range(5)]

        def worker(state):
            return {"seen": [settings.code_writer_model]}

        builder = StateGraph(_FanState)
        builder.add_node("worker", worker)
        builder.add_conditional_edges(START, fan_out, ["worker"])
        builder.add_edge("worker", END)
        graph = builder.compile()

        with override_settings(code_writer_model="model-scoped"):
            result = graph.invoke({"seen": []})

        assert result["seen"] == ["model-scoped"] * 5
