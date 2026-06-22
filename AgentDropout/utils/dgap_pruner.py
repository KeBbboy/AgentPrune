from __future__ import annotations

import math
import re
import threading
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_DGAP_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE_TRANSFORMER_CACHE = {}
_SENTENCE_TRANSFORMER_LOCK = threading.Lock()


class DGAPContentPruner(torch.nn.Module):
    """Query-aware sentence/sub-sentence message pruner for retained graph edges."""

    def __init__(
        self,
        encoder_model_name: str = DEFAULT_DGAP_ENCODER_MODEL,
        embedding_dim: int = 384,
        hidden_dim: int = 128,
        initial_keep_probability: float = 0.85,
        threshold: float = 0.35,
        temperature: float = 1.0,
        kl_temperature: float = 0.1,
        gate_beta: float = 0.5,
        use_query_gate: bool = True,
        max_keep_rate: float = 1.0,
        teacher_top_ratio: float = 0.45,
        rule_score_weight: float = 0.15,
        min_units: int = 2,
        min_keep_units: int = 1,
        min_tokens: int = 8,
        segment_mode: str = "sentence",
        preserve_answer_line: bool = True,
    ):
        super().__init__()
        if not 0 < float(initial_keep_probability) < 1:
            raise ValueError("DGAP initial keep probability must be in (0, 1).")
        if not 0 <= float(threshold) <= 1:
            raise ValueError("DGAP threshold must be in [0, 1].")
        if float(temperature) <= 0:
            raise ValueError("DGAP temperature must be positive.")
        if float(kl_temperature) <= 0:
            raise ValueError("DGAP KL temperature must be positive.")
        if not 0 < float(max_keep_rate) <= 1:
            raise ValueError("DGAP max_keep_rate must be in (0, 1].")
        if not 0 < float(teacher_top_ratio) <= 1:
            raise ValueError("DGAP teacher_top_ratio must be in (0, 1].")
        if float(rule_score_weight) < 0:
            raise ValueError("DGAP rule_score_weight must be non-negative.")
        if segment_mode not in {"sentence", "clause", "line"}:
            raise ValueError("DGAP segment_mode must be one of: sentence, clause, line.")

        self.encoder_model_name = encoder_model_name
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.initial_keep_probability = float(initial_keep_probability)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.kl_temperature = float(kl_temperature)
        self.gate_beta = float(gate_beta)
        self.use_query_gate = bool(use_query_gate)
        self.max_keep_rate = float(max_keep_rate)
        self.teacher_top_ratio = float(teacher_top_ratio)
        self.rule_score_weight = float(rule_score_weight)
        self.min_units = max(1, int(min_units))
        self.min_keep_units = max(0, int(min_keep_units))
        self.min_tokens = max(0, int(min_tokens))
        self.segment_mode = segment_mode
        self.preserve_answer_line = bool(preserve_answer_line)

        self.content_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 4, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, max(1, self.hidden_dim // 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(1, self.hidden_dim // 2), 1),
        )
        self._init_parameters()
        self.reset_run_state()
        self.num_seen = 0
        self.num_pruned = 0
        self.original_tokens = 0
        self.pruned_tokens = 0

    def _init_parameters(self):
        final_layer = self.content_mlp[-1]
        if isinstance(final_layer, torch.nn.Linear):
            torch.nn.init.zeros_(final_layer.weight)
            keep_logit = math.log(self.initial_keep_probability / (1 - self.initial_keep_probability))
            torch.nn.init.constant_(final_layer.bias, keep_logit)

    def reset_run_state(self):
        self._log_probs: list[torch.Tensor] = []
        self._expected_keep_losses: list[torch.Tensor] = []
        self._semantic_losses: list[torch.Tensor] = []
        self._selection_losses: list[torch.Tensor] = []
        self._policy_decision_count = 0

    def share_parameters(self, source: "DGAPContentPruner"):
        self.content_mlp = source.content_mlp
        self.gate_mlp = source.gate_mlp

    def prune(
        self,
        text: Any,
        query: Any,
        domain: str = "",
        sender_role: str = "",
        receiver_role: str = "",
        sender_id: str = "",
        receiver_id: str = "",
        source: str = "spatial",
        message_type: str = "full",
    ) -> Any:
        if not isinstance(text, str):
            return text

        stripped_text = text.strip()
        if not stripped_text:
            return text

        self.num_seen += 1
        original_token_count = self._approx_token_count(stripped_text)
        if original_token_count < self.min_tokens:
            return text

        answer_line, compressible_text = self._split_leading_answer_line(stripped_text, domain)
        if not self.preserve_answer_line:
            answer_line = None
            compressible_text = stripped_text
        compressible_text = compressible_text.strip()
        if not compressible_text:
            return text

        units = self._split_units(compressible_text)
        if len(units) < self.min_units:
            return text

        query_text = self._query_to_text(query)
        sender_text = sender_role or sender_id or "source agent"
        receiver_text = receiver_role or receiver_id or "target agent"
        embeddings = self._encode([sender_text, receiver_text, query_text, *units])
        sender_emb, receiver_emb, query_emb = embeddings[0], embeddings[1], embeddings[2]
        unit_embs = embeddings[3:]
        unit_count = unit_embs.shape[0]

        features = torch.cat(
            [
                sender_emb.expand(unit_count, -1),
                receiver_emb.expand(unit_count, -1),
                unit_embs,
                query_emb.expand(unit_count, -1),
            ],
            dim=-1,
        )
        logits = self.content_mlp(features).squeeze(-1)
        temperature = self._temperature_from_query_embedding(query_emb)
        probs = torch.sigmoid(logits / temperature).clamp(1e-6, 1 - 1e-6)
        rule_scores = self._rule_importance_scores(units, unit_embs)
        self._expected_keep_losses.append(probs.mean())
        self._semantic_losses.append(self._semantic_preservation_loss(unit_embs, probs))
        self._selection_losses.append(
            self._teacher_selection_loss(unit_embs, query_emb, probs, units, rule_scores)
        )

        if self.training:
            hard_mask = (torch.rand_like(probs) < probs).float()
            log_prob = (
                hard_mask * torch.log(probs)
                + (1 - hard_mask) * torch.log(1 - probs)
            ).sum()
            self._log_probs.append(log_prob)
            self._policy_decision_count += int(unit_count)
        else:
            hard_mask = (probs >= self.threshold).float()

        output_scores = probs + self.rule_score_weight * rule_scores
        output_mask = hard_mask.clone()
        min_keep_units = min(self.min_keep_units, unit_count)
        if output_mask.sum().item() < min_keep_units:
            keep_indices = torch.topk(output_scores, k=min_keep_units).indices
            output_mask[keep_indices] = 1.0
        elif output_mask.sum().item() == 0 and answer_line is None:
            output_mask[torch.argmax(output_scores).item()] = 1.0

        max_keep_units = max(min_keep_units, int(math.ceil(self.max_keep_rate * unit_count)))
        if output_mask.sum().item() > max_keep_units:
            keep_indices = torch.topk(output_scores, k=max_keep_units).indices
            output_mask = torch.zeros_like(output_mask)
            output_mask[keep_indices] = 1.0

        kept_units = [unit for unit, keep in zip(units, output_mask.detach().cpu().tolist()) if keep >= 0.5]
        pruned_body = " ".join(kept_units).strip()
        pruned_text = self._restore_leading_answer_line(pruned_body, answer_line)
        if not pruned_text.strip():
            return text

        pruned_token_count = self._approx_token_count(pruned_text)
        if pruned_token_count < original_token_count:
            self.num_pruned += 1
            self.original_tokens += original_token_count
            self.pruned_tokens += pruned_token_count
            return pruned_text
        return text

    def log_prob(self) -> torch.Tensor:
        if not self._log_probs:
            return self._zero_tensor()
        return torch.stack(self._log_probs).sum()

    def normalized_log_prob(self) -> torch.Tensor:
        return self.log_prob() / max(self._policy_decision_count, 1)

    def regularization_loss(self) -> torch.Tensor:
        if not self._expected_keep_losses:
            return self._zero_tensor()
        return torch.stack(self._expected_keep_losses).mean()

    def semantic_preservation_loss(self) -> torch.Tensor:
        if not self._semantic_losses:
            return self._zero_tensor()
        return torch.stack(self._semantic_losses).mean()

    def selection_loss(self) -> torch.Tensor:
        if not self._selection_losses:
            return self._zero_tensor()
        return torch.stack(self._selection_losses).mean()

    def query_temperature(self, query: Any) -> torch.Tensor:
        query_emb = self._encode([self._query_to_text(query)])[0]
        return self._temperature_from_query_embedding(query_emb)

    def summary(self) -> str:
        if self.num_pruned == 0:
            return "DGAP content pruned 0 messages."
        ratio = self.pruned_tokens / max(self.original_tokens, 1)
        return (
            f"DGAP content pruned {self.num_pruned}/{self.num_seen} messages "
            f"from {self.original_tokens} to {self.pruned_tokens} approx tokens "
            f"(keep ratio {ratio:.3f})."
        )

    def _zero_tensor(self) -> torch.Tensor:
        parameter = next(self.parameters(), None)
        if parameter is None:
            return torch.tensor(0.0)
        return parameter.sum() * 0.0

    def _semantic_preservation_loss(self, unit_embs: torch.Tensor, keep_probs: torch.Tensor) -> torch.Tensor:
        full_summary = unit_embs.mean(dim=0)
        keep_weights = keep_probs.unsqueeze(-1)
        soft_pruned_summary = (keep_weights * unit_embs).sum(dim=0) / keep_weights.sum().clamp_min(1e-6)
        kl_temperature = torch.tensor(
            self.kl_temperature,
            dtype=unit_embs.dtype,
            device=unit_embs.device,
        )
        original_distribution = F.softmax(full_summary / kl_temperature, dim=-1)
        pruned_log_distribution = F.log_softmax(soft_pruned_summary / kl_temperature, dim=-1)
        return F.kl_div(
            pruned_log_distribution,
            original_distribution,
            reduction="sum",
        ) * (kl_temperature ** 2)

    def _teacher_selection_loss(
        self,
        unit_embs: torch.Tensor,
        query_emb: torch.Tensor,
        keep_probs: torch.Tensor,
        units: list[str],
        rule_scores: torch.Tensor,
    ) -> torch.Tensor:
        targets = self._teacher_keep_targets(unit_embs, query_emb, units, rule_scores)
        return F.binary_cross_entropy(keep_probs, targets, reduction="mean")

    def _teacher_keep_targets(
        self,
        unit_embs: torch.Tensor,
        query_emb: torch.Tensor,
        units: list[str],
        rule_scores: torch.Tensor,
    ) -> torch.Tensor:
        unit_count = unit_embs.shape[0]
        query_scores = F.cosine_similarity(
            unit_embs,
            query_emb.unsqueeze(0).expand(unit_count, -1),
            dim=-1,
        )
        query_scores = (query_scores + 1.0) * 0.5
        contribution_scores = self._summary_contribution_scores(unit_embs)
        teacher_scores = (
            0.45 * self._normalize_scores(query_scores)
            + 0.35 * self._normalize_scores(contribution_scores)
            + 0.20 * rule_scores
        )
        teacher_scores = self._normalize_scores(teacher_scores)
        top_k = max(self.min_keep_units, int(math.ceil(self.teacher_top_ratio * unit_count)))
        top_k = min(unit_count, max(1, top_k))
        top_indices = torch.topk(teacher_scores, k=top_k).indices

        targets = (0.05 + 0.25 * teacher_scores).clamp(max=0.95)
        targets[top_indices] = 1.0
        return targets.detach()

    def _summary_contribution_scores(self, unit_embs: torch.Tensor) -> torch.Tensor:
        unit_count = unit_embs.shape[0]
        if unit_count <= 1:
            return torch.ones(unit_count, dtype=unit_embs.dtype, device=unit_embs.device)
        full_summary = unit_embs.mean(dim=0)
        leave_one_out = (unit_embs.sum(dim=0, keepdim=True) - unit_embs) / (unit_count - 1)
        kl_temperature = torch.tensor(
            self.kl_temperature,
            dtype=unit_embs.dtype,
            device=unit_embs.device,
        )
        original_distribution = F.softmax(full_summary / kl_temperature, dim=-1).unsqueeze(0)
        leave_one_out_log_distribution = F.log_softmax(leave_one_out / kl_temperature, dim=-1)
        return F.kl_div(
            leave_one_out_log_distribution,
            original_distribution.expand_as(leave_one_out_log_distribution),
            reduction="none",
        ).sum(dim=-1) * (kl_temperature ** 2)

    def _rule_importance_scores(self, units: list[str], unit_embs: torch.Tensor) -> torch.Tensor:
        scores = []
        for unit in units:
            text = str(unit)
            lower = text.lower()
            score = 0.0
            if re.search(r"\b(option|choice|answer)\s*[A-J]\b", lower):
                score = max(score, 0.9)
            if re.search(r"\b(correct|incorrect|answer|therefore|thus|hence|consequently)\b", lower):
                score = max(score, 0.8)
            if re.search(r"\b(not|except|least|most|only|all|none|always|never)\b", lower):
                score = max(score, 0.7)
            if re.search(r"\\frac|\\sqrt|[$=<>^*/]|\d", text):
                score = max(score, 0.6)
            if re.search(r"\b(because|since|due to|however|whereas|although|implies|causes|leads to)\b", lower):
                score = max(score, 0.55)
            scores.append(score)
        return torch.tensor(scores, dtype=unit_embs.dtype, device=unit_embs.device)

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if torch.isclose(max_score, min_score):
            return torch.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score).clamp_min(1e-6)

    def _temperature_from_query_embedding(self, query_emb: torch.Tensor) -> torch.Tensor:
        base_temperature = torch.tensor(
            self.temperature,
            dtype=query_emb.dtype,
            device=query_emb.device,
        )
        if not self.use_query_gate:
            return base_temperature
        gate_value = torch.tanh(self.gate_mlp(query_emb).squeeze())
        return torch.clamp(base_temperature + self.gate_beta * gate_value, min=0.05)

    def _encode(self, texts: list[str]) -> torch.Tensor:
        model = _load_sentence_transformer(self.encoder_model_name)
        try:
            encoded = model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        except TypeError:
            encoded = model.encode(texts)
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.float32)
        parameter = next(self.parameters(), None)
        if parameter is not None:
            encoded = encoded.to(device=parameter.device, dtype=parameter.dtype)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        if encoded.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"DGAP encoder produced dimension {encoded.shape[-1]}, "
                f"but dgap_embedding_dim is {self.embedding_dim}."
            )
        return encoded

    def _query_to_text(self, query: Any) -> str:
        if isinstance(query, dict):
            task = query.get("task")
            if task is not None:
                return str(task)
        return str(query)

    def _split_leading_answer_line(self, text: str, domain: str) -> tuple[str | None, str]:
        if domain not in {"mmlu", "mmlu_pro"}:
            return None, text
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^([A-J])(?:$|[\s).:](.*)$)", stripped)
            if not match:
                break
            answer_line = match.group(1)
            first_line_remainder = (match.group(2) or "").lstrip(").: ").strip()
            remaining_lines = lines[idx + 1:]
            compressible_lines = []
            if first_line_remainder:
                compressible_lines.append(first_line_remainder)
            compressible_lines.extend(remaining_lines)
            return answer_line, "\n".join(compressible_lines)
        return None, text

    def _restore_leading_answer_line(self, text: str, answer_line: str | None) -> str:
        if answer_line is None:
            return text
        return f"{answer_line}\n{text}" if text else answer_line

    def _split_units(self, text: str) -> list[str]:
        if self.segment_mode == "line":
            units = text.splitlines()
        else:
            units = re.split(r"(?<=[.!?])\s+|\n+", text)
            if self.segment_mode == "clause":
                clause_units = []
                for unit in units:
                    clause_units.extend(re.split(r"(?<=[;:])\s+|,\s+(?=\w)", unit))
                units = clause_units
        return [unit.strip() for unit in units if unit.strip()]

    def _approx_token_count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


def _load_sentence_transformer(model_name: str):
    if model_name in _SENTENCE_TRANSFORMER_CACHE:
        return _SENTENCE_TRANSFORMER_CACHE[model_name]
    with _SENTENCE_TRANSFORMER_LOCK:
        if model_name in _SENTENCE_TRANSFORMER_CACHE:
            return _SENTENCE_TRANSFORMER_CACHE[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "DGAP content pruning requires `sentence-transformers`. "
                "Install project requirements before enabling --dgap_content_pruning."
            ) from exc
        model = SentenceTransformer(model_name)
        _SENTENCE_TRANSFORMER_CACHE[model_name] = model
        return model


def add_dgap_args(parser):
    parser.add_argument(
        "--dgap_content_pruning",
        action="store_true",
        help="Enable DGAP query-aware content pruning on retained inter-agent messages.",
    )
    parser.add_argument(
        "--dgap_encoder_model_name",
        type=str,
        default=DEFAULT_DGAP_ENCODER_MODEL,
        help="SentenceTransformer encoder used by DGAP.",
    )
    parser.add_argument(
        "--dgap_embedding_dim",
        type=int,
        default=384,
        help="DGAP encoder embedding dimension. all-MiniLM-L6-v2 uses 384.",
    )
    parser.add_argument(
        "--dgap_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for DGAP content and query-gate MLPs.",
    )
    parser.add_argument(
        "--dgap_initial_keep_prob",
        type=float,
        default=0.85,
        help="Initial expected keep probability for DGAP content units.",
    )
    parser.add_argument(
        "--dgap_threshold",
        type=float,
        default=0.35,
        help="Deterministic inference threshold for DGAP content units.",
    )
    parser.add_argument(
        "--dgap_temperature",
        type=float,
        default=1.0,
        help="Base DGAP sampling temperature.",
    )
    parser.add_argument(
        "--dgap_kl_temperature",
        type=float,
        default=0.1,
        help="Softmax temperature used by DGAP KL semantic preservation loss.",
    )
    parser.add_argument(
        "--dgap_gate_beta",
        type=float,
        default=0.5,
        help="Strength of the DGAP query-adaptive temperature gate.",
    )
    parser.add_argument(
        "--dgap_no_query_gate",
        action="store_true",
        help="Disable the lightweight DGAP query-adaptive gate.",
    )
    parser.add_argument(
        "--dgap_gate_edges",
        action="store_true",
        help="Also use DGAP query temperature for optimized edge sampling.",
    )
    parser.add_argument(
        "--dgap_max_keep_rate",
        type=float,
        default=1.0,
        help="Hard maximum fraction of reasoning units kept per message by DGAP.",
    )
    parser.add_argument(
        "--dgap_teacher_top_ratio",
        type=float,
        default=0.45,
        help="Fraction of reasoning units treated as pseudo-teacher positives for DGAP selection loss.",
    )
    parser.add_argument(
        "--dgap_rule_score_weight",
        type=float,
        default=0.15,
        help="Inference-time weight for rule-protected DGAP units such as answers, options, numbers, and conclusions.",
    )
    parser.add_argument(
        "--dgap_min_units",
        type=int,
        default=2,
        help="Skip DGAP content pruning for messages with fewer units.",
    )
    parser.add_argument(
        "--dgap_min_keep_units",
        type=int,
        default=1,
        help="Minimum number of reasoning units DGAP keeps on each retained message edge.",
    )
    parser.add_argument(
        "--dgap_min_tokens",
        type=int,
        default=8,
        help="Skip DGAP content pruning for shorter messages.",
    )
    parser.add_argument(
        "--dgap_segment_mode",
        type=str,
        choices=("sentence", "clause", "line"),
        default="sentence",
        help="Semantic unit granularity for DGAP content pruning.",
    )
    parser.add_argument(
        "--dgap_scope",
        type=str,
        choices=("agent", "final", "all"),
        default="all",
        help="Where DGAP prunes messages: agent edges, final-decision edges, or both.",
    )
    parser.add_argument(
        "--dgap_content_lambda",
        type=float,
        default=0.002,
        help="Regularization weight for expected DGAP content keep rate.",
    )
    parser.add_argument(
        "--dgap_target_keep_rate",
        type=float,
        default=1.0,
        help="Target maximum expected DGAP content keep rate used by the budget penalty.",
    )
    parser.add_argument(
        "--dgap_budget_lambda",
        type=float,
        default=0.0,
        help="Penalty weight for expected DGAP content keep rate above --dgap_target_keep_rate.",
    )
    parser.add_argument(
        "--dgap_lr",
        type=float,
        default=0.001,
        help="Learning rate for DGAP content and query-gate MLP parameters.",
    )
    parser.add_argument(
        "--dgap_task_lambda",
        type=float,
        default=0.02,
        help="Policy-gradient weight that updates DGAP sentence masks from final task utility.",
    )
    parser.add_argument(
        "--dgap_reward_baseline",
        type=float,
        default=0.5,
        help="Baseline subtracted from task utility for DGAP content policy-gradient updates.",
    )
    parser.add_argument(
        "--dgap_semantic_lambda",
        type=float,
        default=1.0,
        help="Weight for semantic preservation between original and soft-pruned reasoning embeddings.",
    )
    parser.add_argument(
        "--dgap_selection_lambda",
        type=float,
        default=0.0,
        help="Weight for query-aware pseudo-teacher sentence selection loss.",
    )


def dgap_graph_kwargs(args) -> dict[str, Any]:
    return {
        "dgap_content_pruning": getattr(args, "dgap_content_pruning", False),
        "dgap_encoder_model_name": getattr(args, "dgap_encoder_model_name", DEFAULT_DGAP_ENCODER_MODEL),
        "dgap_embedding_dim": getattr(args, "dgap_embedding_dim", 384),
        "dgap_hidden_dim": getattr(args, "dgap_hidden_dim", 128),
        "dgap_initial_keep_prob": getattr(args, "dgap_initial_keep_prob", 0.85),
        "dgap_threshold": getattr(args, "dgap_threshold", 0.35),
        "dgap_temperature": getattr(args, "dgap_temperature", 1.0),
        "dgap_kl_temperature": getattr(args, "dgap_kl_temperature", 0.1),
        "dgap_gate_beta": getattr(args, "dgap_gate_beta", 0.5),
        "dgap_use_query_gate": not getattr(args, "dgap_no_query_gate", False),
        "dgap_max_keep_rate": getattr(args, "dgap_max_keep_rate", 1.0),
        "dgap_teacher_top_ratio": getattr(args, "dgap_teacher_top_ratio", 0.45),
        "dgap_rule_score_weight": getattr(args, "dgap_rule_score_weight", 0.15),
        "dgap_gate_edges": getattr(args, "dgap_gate_edges", False),
        "dgap_min_units": getattr(args, "dgap_min_units", 2),
        "dgap_min_keep_units": getattr(args, "dgap_min_keep_units", 1),
        "dgap_min_tokens": getattr(args, "dgap_min_tokens", 8),
        "dgap_segment_mode": getattr(args, "dgap_segment_mode", "sentence"),
        "dgap_scope": getattr(args, "dgap_scope", "all"),
    }
