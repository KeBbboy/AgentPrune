import torch

from AgentDropout.utils import dgap_pruner
from AgentDropout.utils.dgap_pruner import DGAPContentPruner


class FakeSentenceEncoder:
    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            base = float(sum(ord(ch) for ch in str(text)) % 17)
            rows.append(torch.full((self.dim,), base / 17.0))
        return torch.stack(rows)


def setup_fake_encoder(model_name="fake-mini", dim=384):
    dgap_pruner._SENTENCE_TRANSFORMER_CACHE.clear()
    dgap_pruner._SENTENCE_TRANSFORMER_CACHE[model_name] = FakeSentenceEncoder(dim=dim)
    return model_name


def test_dgap_pruner_preserves_mmlu_answer_line_when_reasoning_is_removed():
    model_name = setup_fake_encoder()
    pruner = DGAPContentPruner(
        encoder_model_name=model_name,
        threshold=0.5,
        min_tokens=0,
        min_units=1,
        min_keep_units=0,
    )
    pruner.eval()
    with torch.no_grad():
        pruner.content_mlp[-1].bias.fill_(-10.0)

    result = pruner.prune(
        "C\nFine argues that both A and B apply. A second reason is weaker.",
        query={"task": "Which option is correct?"},
        domain="mmlu",
        sender_role="Critic",
        receiver_role="Doctor",
    )

    assert result == "C"


def test_dgap_pruner_keeps_minimum_reasoning_units():
    model_name = setup_fake_encoder()
    pruner = DGAPContentPruner(
        encoder_model_name=model_name,
        threshold=0.5,
        min_tokens=0,
        min_units=1,
        min_keep_units=1,
    )
    pruner.eval()
    with torch.no_grad():
        pruner.content_mlp[-1].bias.fill_(-10.0)

    result = pruner.prune(
        "C\nFirst reason. Second reason.",
        query={"task": "Which option is correct?"},
        domain="mmlu",
        sender_role="Critic",
        receiver_role="Doctor",
    )

    assert result.startswith("C\n")
    assert result.count("reason") == 1


def test_dgap_pruner_exposes_trainable_log_prob_and_regularization():
    model_name = setup_fake_encoder()
    pruner = DGAPContentPruner(
        encoder_model_name=model_name,
        initial_keep_probability=0.5,
        min_tokens=0,
        min_units=1,
    )
    pruner.train()

    pruner.prune(
        "A\nFirst useful reason. Second useful reason.",
        query={"task": "Question with options."},
        domain="mmlu",
        sender_role="Solver",
        receiver_role="Critic",
    )

    loss = -pruner.log_prob() + pruner.regularization_loss()
    loss.backward()

    assert pruner.log_prob().requires_grad
    assert pruner.regularization_loss().requires_grad
    assert any(param.grad is not None for param in pruner.parameters())


def test_dgap_pruner_semantic_preservation_loss_is_trainable():
    model_name = setup_fake_encoder()
    pruner = DGAPContentPruner(
        encoder_model_name=model_name,
        initial_keep_probability=0.5,
        min_tokens=0,
        min_units=1,
    )
    pruner.train()

    pruner.prune(
        "A\nFirst useful reason. Second useful reason.",
        query={"task": "Question with options."},
        domain="mmlu",
        sender_role="Solver",
        receiver_role="Critic",
    )

    loss = pruner.semantic_preservation_loss() + 0.01 * pruner.regularization_loss()
    loss.backward()

    assert pruner.semantic_preservation_loss().requires_grad
    assert any(param.grad is not None for param in pruner.parameters())


def test_dgap_pruner_selection_loss_is_trainable():
    model_name = setup_fake_encoder()
    pruner = DGAPContentPruner(
        encoder_model_name=model_name,
        initial_keep_probability=0.5,
        min_tokens=0,
        min_units=1,
        teacher_top_ratio=0.5,
    )
    pruner.train()

    pruner.prune(
        "A\nOption A is correct because it preserves the key condition. "
        "Option B ignores the numerical constraint. Therefore answer A follows.",
        query={"task": "Which option follows from the numerical condition?"},
        domain="mmlu",
        sender_role="Solver",
        receiver_role="Critic",
    )

    loss = pruner.selection_loss()
    loss.backward()

    assert pruner.selection_loss().requires_grad
    assert any(param.grad is not None for param in pruner.parameters())


def test_dgap_pruner_hard_max_keep_rate_limits_units():
    model_name = setup_fake_encoder()
    pruner = DGAPContentPruner(
        encoder_model_name=model_name,
        initial_keep_probability=0.99,
        threshold=0.01,
        min_tokens=0,
        min_units=1,
        min_keep_units=1,
        max_keep_rate=0.5,
    )
    pruner.eval()

    result = pruner.prune(
        "A\nFirst reason. Second reason. Third reason. Fourth reason.",
        query={"task": "Which answer is supported?"},
        domain="mmlu",
        sender_role="Solver",
        receiver_role="Critic",
    )

    assert result.startswith("A\n")
    assert result.count("reason") <= 2
