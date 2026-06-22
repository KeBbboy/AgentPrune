from dataset.mmlu_dataset import MMLUDataset
from AgentDropout.prompt.mmlu_prompt_set import MMLUPromptSet


def test_mmlu_postprocess_preserves_first_line_answer_over_later_phrases():
    answer = (
        "A\n\nAll agents selected A. "
        "The correct answer is more likely to be none of the options listed."
    )

    assert MMLUDataset._extract_option_answer(answer) == "A"
    assert MMLUPromptSet._extract_option_answer(answer) == "A"


def test_mmlu_postprocess_handles_markdown_final_answer():
    answer = (
        "The propositions are evaluated below.\n\n"
        "**Option B** is inconsistent.\n"
        "**Option C** is inconsistent.\n"
        "Thus, the correct answer is **D**."
    )

    assert MMLUDataset._extract_option_answer(answer) == "D"
    assert MMLUPromptSet._extract_option_answer(answer) == "D"


def test_mmlu_postprocess_avoids_markdown_star_as_answer():
    answer = "B\n\nThus, the correct answer is **B**."

    assert MMLUDataset._extract_option_answer(answer) == "B"
    assert MMLUPromptSet._extract_option_answer(answer) == "B"


def test_mmlu_postprocess_prefers_explicit_final_answer_when_first_line_conflicts():
    answer = "C\n\nAfter checking the propositions, the correct answer is **D**."

    assert MMLUDataset._extract_option_answer(answer) == "D"
    assert MMLUPromptSet._extract_option_answer(answer) == "D"
