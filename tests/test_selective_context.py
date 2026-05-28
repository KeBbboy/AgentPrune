import re

from AgentDropout.context.selective_context import (
    SelectiveContextCompressor,
    SelectiveContextConfig,
    TokenScore,
    selective_context_disabled,
)


class FakeScorer:
    def score(self, text):
        token_scores = []
        for match in re.finditer(r"\S+", text):
            token = match.group(0).lower()
            score = 10.0 if "rare" in token or "insight" in token else 1.0
            token_scores.append(TokenScore(match.start(), match.end(), score))
        return token_scores


def test_selective_context_keeps_high_surprisal_and_protected_units():
    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(
            enabled=True,
            reduce_ratio=0.4,
            granularity="sentence",
            min_chars=0,
            min_units=1,
            preserve_first_line=False,
            preserve_first_units=1,
            preserve_last_units=1,
        ),
        scorer=FakeScorer(),
    )
    text = (
        "First answer should stay. "
        "Boring middle sentence one. "
        "Rare insight sentence should stay. "
        "Boring middle sentence two. "
        "Final answer should stay."
    )

    compressed = compressor.compress(text)

    assert "First answer should stay." in compressed
    assert "Rare insight sentence should stay." in compressed
    assert "Final answer should stay." in compressed
    assert "Boring middle sentence one." not in compressed
    assert "Boring middle sentence two." not in compressed


def test_selective_context_compresses_string_lists():
    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(
            enabled=True,
            reduce_ratio=0.5,
            granularity="sentence",
            min_chars=0,
            min_units=1,
            preserve_first_line=False,
            preserve_first_units=0,
            preserve_last_units=0,
        ),
        scorer=FakeScorer(),
    )
    payload = [
        "Boring sentence. Rare insight should stay.",
        "Another boring sentence. Rare detail should stay.",
    ]

    compressed = compressor.compress_payload(payload)

    assert "Rare insight should stay." in compressed[0]
    assert "Rare detail should stay." in compressed[1]


def test_selective_context_disabled_returns_original_without_scoring():
    class ExplodingScorer:
        def score(self, text):
            raise AssertionError("disabled compressor should not score")

    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(enabled=False),
        scorer=ExplodingScorer(),
    )

    assert compressor.compress("unchanged text") == "unchanged text"


def test_selective_context_runtime_disable_returns_original_without_scoring():
    class ExplodingScorer:
        def score(self, text):
            raise AssertionError("runtime-disabled compressor should not score")

    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(enabled=True),
        scorer=ExplodingScorer(),
    )

    with selective_context_disabled():
        assert compressor.compress("unchanged text") == "unchanged text"


def test_selective_context_preserves_first_line_answer():
    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(
            enabled=True,
            reduce_ratio=0.5,
            granularity="sentence",
            min_chars=0,
            min_units=1,
            preserve_first_line=True,
            preserve_first_units=0,
            preserve_last_units=0,
        ),
        scorer=FakeScorer(),
    )
    text = (
        "Answer: B\n"
        "Boring middle sentence one. "
        "Rare insight sentence should stay. "
        "Boring middle sentence two."
    )

    compressed = compressor.compress(text)

    assert compressed.startswith("Answer: B\n")
    assert "Rare insight sentence should stay." in compressed
    assert "Boring middle sentence one." not in compressed
