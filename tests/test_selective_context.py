import re

from AgentDropout.context.selective_context import (
    SelectiveContext,
    SelectiveContextCompressor,
    SelectiveContextConfig,
    TokenScore,
    selective_context_disabled,
)


class FakeScorer:
    def score(self, text):
        token_scores = []
        for match in re.finditer(r"\S+", text):
            token = re.sub(r"[^a-z0-9]+", "", match.group(0).lower())
            if token in {"rare", "insight", "detail"}:
                score = 10.0
            elif token == "verylow":
                score = 0.1
            elif token == "low":
                score = 0.2
            else:
                score = 1.0
            token_scores.append(TokenScore(match.start(), match.end(), score))
        return token_scores


def test_selective_context_default_model_is_qwen():
    assert SelectiveContextConfig().model_name == "Qwen/Qwen2.5-0.5B"


def test_official_style_call_returns_context_and_reduced_content():
    sc = SelectiveContext(model_type="Qwen/Qwen2.5-0.5B", lang="en", scorer=FakeScorer())
    text = (
        "verylow verylow. "
        "Rare insight sentence should stay. "
        "low low."
    )

    context, reduced_content = sc(text, reduce_ratio=0.5, reduce_level="sent")

    assert "Rare insight sentence should stay." in context
    assert "verylow verylow." not in context
    assert any("verylow verylow." in unit for unit in reduced_content)


def test_phrase_split_uses_official_style_spacy_tokens(monkeypatch):
    class FakeToken:
        def __init__(self, text, idx, whitespace):
            self.text = text
            self.idx = idx
            self.whitespace_ = whitespace

    class FakeDoc(list):
        pass

    sc = SelectiveContext(model_type="Qwen/Qwen2.5-0.5B", lang="en", scorer=FakeScorer())

    def fake_phrase_tokenizer():
        return lambda text: FakeDoc(
            [
                FakeToken("The rare phrase", 0, " "),
                FakeToken("low", 16, " "),
                FakeToken("detail", 20, ""),
            ]
        )

    monkeypatch.setattr(sc, "_get_phrase_tokenizer", fake_phrase_tokenizer)

    context, reduced_content = sc(
        "The rare phrase low detail",
        reduce_ratio=0.5,
        reduce_level="phrase",
    )

    assert "The rare phrase" in context
    assert "detail" in context
    assert "low" not in context
    assert reduced_content == ["low"]


def test_official_style_reserved_context_ratio_keeps_tail_uncompressed():
    sc = SelectiveContext(model_type="Qwen/Qwen2.5-0.5B", lang="en", scorer=FakeScorer())
    text = (
        "verylow verylow. "
        "Rare insight sentence should stay. "
        "Reserved tail should remain untouched."
    )

    context, reduced_content = sc(
        text,
        reduce_ratio=0.5,
        reduce_level="sent",
        reserved_context_ratio=0.5,
    )

    assert "Reserved tail should remain untouched." in context
    assert reduced_content


def test_selective_context_keeps_high_surprisal_and_protected_units():
    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(
            enabled=True,
            reduce_ratio=0.4,
            granularity="sent",
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
        "verylow verylow. "
        "Rare insight sentence should stay. "
        "low low. "
        "Final answer should stay."
    )

    compressed = compressor.compress(text)

    assert "First answer should stay." in compressed
    assert "Rare insight sentence should stay." in compressed
    assert "Final answer should stay." in compressed
    assert "verylow verylow." not in compressed
    assert "low low." not in compressed


def test_selective_context_compresses_string_lists():
    compressor = SelectiveContextCompressor(
        SelectiveContextConfig(
            enabled=True,
            reduce_ratio=0.5,
            granularity="sent",
            min_chars=0,
            min_units=1,
            preserve_first_line=False,
            preserve_first_units=0,
            preserve_last_units=0,
        ),
        scorer=FakeScorer(),
    )
    payload = [
        "verylow verylow. Rare insight should stay.",
        "low low. Rare detail should stay.",
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
            granularity="sent",
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
        "verylow verylow. "
        "Rare insight sentence should stay. "
        "low low."
    )

    compressed = compressor.compress(text)

    assert compressed.startswith("Answer: B\n")
    assert "Rare insight sentence should stay." in compressed
    assert "verylow verylow." not in compressed
