"""Tests for separating Claude's reasoning from its visible output.

With adaptive thinking enabled a message's ``content`` is a list of blocks
mixing reasoning with the generated script. If a reasoning block ever reached
``candidate_code`` it would be written verbatim into the generated ``.py``
file, and every downstream syntax check would fail. These tests pin that
boundary against each block shape we can be handed.
"""

from honegumi_rag_assistant.nodes.code_writer import extract_text, extract_reasoning


# Raw Anthropic block shapes, as ChatAnthropic populates `.content`.
RAW_THINKING = {"type": "thinking", "thinking": "First I should replace branin.", "index": 0}
RAW_TEXT = {"type": "text", "text": "import numpy as np\n", "index": 1}

# LangChain's normalised `.content_blocks` shapes.
STD_REASONING = {"type": "reasoning", "reasoning": "Consider the constraints."}
STD_TEXT = {"type": "text", "text": "ax_client = AxClient()\n"}

# A raw block LangChain did not recognise gets wrapped.
NON_STANDARD_THINKING = {"type": "non_standard", "value": RAW_THINKING}
NON_STANDARD_TEXT = {"type": "non_standard", "value": RAW_TEXT}


class TestExtractText:
    def test_plain_string_passes_through(self):
        """Thinking disabled: content is a bare string."""
        assert extract_text("print('hi')") == "print('hi')"

    def test_picks_text_out_of_mixed_raw_blocks(self):
        assert extract_text([RAW_THINKING, RAW_TEXT]) == "import numpy as np\n"

    def test_picks_text_out_of_mixed_standard_blocks(self):
        assert extract_text([STD_REASONING, STD_TEXT]) == "ax_client = AxClient()\n"

    def test_unwraps_non_standard_text(self):
        assert extract_text([NON_STANDARD_THINKING, NON_STANDARD_TEXT]) == "import numpy as np\n"

    def test_concatenates_multiple_text_blocks_in_order(self):
        blocks = [
            {"type": "text", "text": "line one\n"},
            RAW_THINKING,
            {"type": "text", "text": "line two\n"},
        ]
        assert extract_text(blocks) == "line one\nline two\n"

    def test_reasoning_only_yields_empty_string(self):
        assert extract_text([RAW_THINKING, STD_REASONING]) == ""

    def test_redacted_thinking_is_ignored(self):
        blocks = [{"type": "redacted_thinking", "data": "encrypted"}, RAW_TEXT]
        assert extract_text(blocks) == "import numpy as np\n"

    def test_empty_and_none_content(self):
        assert extract_text([]) == ""
        assert extract_text(None) == ""

    def test_bare_string_inside_block_list(self):
        assert extract_text(["a", "b"]) == "ab"


class TestExtractReasoning:
    def test_raw_thinking_block(self):
        assert extract_reasoning([RAW_THINKING]) == "First I should replace branin."

    def test_standard_reasoning_block(self):
        assert extract_reasoning([STD_REASONING]) == "Consider the constraints."

    def test_non_standard_wrapper(self):
        assert extract_reasoning([NON_STANDARD_THINKING]) == "First I should replace branin."

    def test_nested_dict_reasoning_value(self):
        blocks = [{"type": "reasoning", "reasoning": {"text": "nested form"}}]
        assert extract_reasoning(blocks) == "nested form"

    def test_text_blocks_yield_no_reasoning(self):
        assert extract_reasoning([RAW_TEXT, STD_TEXT]) == ""

    def test_plain_string_yields_no_reasoning(self):
        """A bare string is visible output, never reasoning."""
        assert extract_reasoning("print('hi')") == ""

    def test_display_omitted_yields_empty(self):
        """display='omitted' still emits thinking blocks, with empty text."""
        assert extract_reasoning([{"type": "thinking", "thinking": ""}]) == ""


class TestNoLeakage:
    """The property that actually matters: the two never overlap."""

    def test_reasoning_never_appears_in_extracted_code(self):
        secret = "SECRET_REASONING_MARKER"
        blocks = [
            {"type": "thinking", "thinking": secret},
            {"type": "reasoning", "reasoning": secret},
            {"type": "non_standard", "value": {"type": "thinking", "thinking": secret}},
            {"type": "text", "text": "def evaluate(x):\n    return x\n"},
        ]
        code = extract_text(blocks)
        assert secret not in code
        assert code == "def evaluate(x):\n    return x\n"
        # ...and the reasoning is still recoverable for debug output.
        assert secret in extract_reasoning(blocks)

    def test_accumulated_stream_stays_clean(self):
        """Simulate a chunk stream: thinking first, then code."""
        chunks = [
            [{"type": "thinking", "thinking": "planning..."}],
            [{"type": "thinking", "thinking": "more planning..."}],
            [{"type": "text", "text": "import ax\n"}],
            [{"type": "text", "text": "print(1)\n"}],
        ]
        code = "".join(extract_text(c) for c in chunks)
        assert code == "import ax\nprint(1)\n"
        assert "planning" not in code
