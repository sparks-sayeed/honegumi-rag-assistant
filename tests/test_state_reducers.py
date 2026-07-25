"""Tests for the custom state reducers.

These reducers are what make the parallel retrieval fan-in work: LangGraph
calls them to merge the partial state returned by each ``Send`` branch. If
``merge_contexts`` replaced instead of accumulated, the Code Writer would only
ever see the contexts from whichever retriever happened to finish last.
"""

from honegumi_rag_assistant.states import merge_contexts, merge_vectorstore_flag


class TestMergeContexts:
    """The contexts reducer must accumulate across parallel branches."""

    def test_accumulates_both_sides(self):
        left = [{"text": "a"}]
        right = [{"text": "b"}]
        assert merge_contexts(left, right) == [{"text": "a"}, {"text": "b"}]

    def test_handles_none_left(self):
        assert merge_contexts(None, [{"text": "b"}]) == [{"text": "b"}]

    def test_handles_none_right(self):
        assert merge_contexts([{"text": "a"}], None) == [{"text": "a"}]

    def test_handles_both_none(self):
        assert merge_contexts(None, None) == []

    def test_handles_empty_lists(self):
        assert merge_contexts([], []) == []

    def test_does_not_mutate_inputs(self):
        left = [{"text": "a"}]
        right = [{"text": "b"}]
        merge_contexts(left, right)
        assert left == [{"text": "a"}]
        assert right == [{"text": "b"}]

    def test_repeated_folding_accumulates(self):
        """LangGraph folds one branch at a time; N branches must yield N items."""
        acc = []
        for i in range(7):
            acc = merge_contexts(acc, [{"text": f"ctx-{i}"}])
        assert len(acc) == 7
        assert [c["text"] for c in acc] == [f"ctx-{i}" for i in range(7)]


class TestMergeVectorstoreFlag:
    """The flag is an OR: any retriever reporting a missing store wins."""

    def test_true_wins_from_right(self):
        assert merge_vectorstore_flag(None, True) is True

    def test_true_wins_from_left(self):
        assert merge_vectorstore_flag(True, None) is True

    def test_all_none_is_false(self):
        assert merge_vectorstore_flag(None, None) is False

    def test_false_and_false(self):
        assert merge_vectorstore_flag(False, False) is False

    def test_true_is_sticky_across_folds(self):
        """A single missing-store report must survive later clean branches."""
        acc = merge_vectorstore_flag(None, True)
        for _ in range(5):
            acc = merge_vectorstore_flag(acc, None)
        assert acc is True
