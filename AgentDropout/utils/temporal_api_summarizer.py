from __future__ import annotations

import re
from collections import Counter
from typing import Any


DEFAULT_TEMPORAL_OPTION_WORD_BUDGET = 0
DEFAULT_TEMPORAL_REASON_WORD_BUDGET = 0
DEFAULT_TEMPORAL_CONFLICT_OPTION_WORD_BUDGET = 12
DEFAULT_TEMPORAL_CONFLICT_REASON_WORD_BUDGET = 18
DEFAULT_TEMPORAL_RELIABLE_MAJORITY_THRESHOLD = 0.75
DEFAULT_TEMPORAL_IGNORED_ROLES: set[str] = set()
DEFAULT_TEMPORAL_VOTE_STATE_ID = "__temporal_vote_state__"
TEMPORAL_SUMMARY_MODES = ("answer_only", "vote_state", "conflict_clue")


class TemporalAPISummarizer:
    """Receiver-scoped compressor for previous-round temporal memory."""

    def __init__(
        self,
        option_word_budget: int | None = None,
        reason_word_budget: int | None = None,
        ignored_sender_roles: set[str] | None = None,
        include_vote_state: bool = True,
        summary_mode: str | None = None,
        reliable_majority_threshold: float = DEFAULT_TEMPORAL_RELIABLE_MAJORITY_THRESHOLD,
        debug: bool = False,
    ):
        if summary_mode is None:
            summary_mode = "vote_state" if include_vote_state else "answer_only"
        if summary_mode not in TEMPORAL_SUMMARY_MODES:
            raise ValueError(
                f"Unknown temporal summary mode: {summary_mode}. "
                f"Choose one of {', '.join(TEMPORAL_SUMMARY_MODES)}."
            )
        if option_word_budget is None:
            option_word_budget = (
                DEFAULT_TEMPORAL_CONFLICT_OPTION_WORD_BUDGET
                if summary_mode == "conflict_clue"
                else DEFAULT_TEMPORAL_OPTION_WORD_BUDGET
            )
        if reason_word_budget is None:
            reason_word_budget = (
                DEFAULT_TEMPORAL_CONFLICT_REASON_WORD_BUDGET
                if summary_mode == "conflict_clue"
                else DEFAULT_TEMPORAL_REASON_WORD_BUDGET
            )
        self.summary_mode = summary_mode
        self.option_word_budget = max(0, int(option_word_budget))
        self.reason_word_budget = max(0, int(reason_word_budget))
        self.reliable_majority_threshold = max(0.5, min(1.0, float(reliable_majority_threshold)))
        self.ignored_sender_roles = (
            set(DEFAULT_TEMPORAL_IGNORED_ROLES)
            if ignored_sender_roles is None
            else set(ignored_sender_roles)
        )
        self.include_vote_state = self.summary_mode in {"vote_state", "conflict_clue"}
        self.debug = debug
        self.num_seen = 0
        self.num_compressed = 0
        self.num_role_filtered = 0
        self.original_words = 0
        self.compressed_words = 0

    def aggregate(
        self,
        temporal_info: dict[str, dict],
        raw_inputs: Any = None,
        domain: str = "",
        receiver_id: str | None = None,
        **kwargs,
    ) -> dict[str, dict]:
        eligible_info = self._eligible_temporal_info(temporal_info)
        if not eligible_info:
            return {}

        self.num_seen += 1
        option_map = self._extract_option_map(raw_inputs)
        if self.summary_mode in {"vote_state", "conflict_clue"}:
            return self._aggregate_vote_state(
                eligible_info=eligible_info,
                option_map=option_map,
                receiver_id=receiver_id,
                raw_inputs=raw_inputs,
                domain=domain,
                include_conflict_clues=self.summary_mode == "conflict_clue",
            )
        return self._aggregate_individual_labels(eligible_info, option_map=option_map)

    def _aggregate_individual_labels(
        self,
        eligible_info: dict[str, dict],
        option_map: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        compressed_info: dict[str, dict] = {}
        for sender_id, info in eligible_info.items():
            output = str(info.get("output", "")).strip()
            compressed_output = self._compress_output(output, option_map=option_map)
            updated_info = dict(info)
            updated_info["output"] = compressed_output
            updated_info["message_type"] = "temporal_edge_compressed"
            updated_info["source_message_type"] = info.get("message_type", "full")
            compressed_info[sender_id] = updated_info
            self.num_compressed += 1
            self.original_words += self._approx_word_count(output)
            self.compressed_words += self._approx_word_count(compressed_output)
        return compressed_info

    def _aggregate_vote_state(
        self,
        eligible_info: dict[str, dict],
        option_map: dict[str, str] | None = None,
        receiver_id: str | None = None,
        raw_inputs: Any = None,
        domain: str = "",
        include_conflict_clues: bool = False,
    ) -> dict[str, dict]:
        parsed_entries = []
        fallback_info: dict[str, dict] = {}
        original_words = 0
        fallback_words = 0

        for sender_id, info in eligible_info.items():
            output = str(info.get("output", "")).strip()
            original_words += self._approx_word_count(output)
            answer = self._extract_answer_label(output)
            if answer:
                parsed_entries.append(
                    {
                        "id": sender_id,
                        "role": str(info.get("role", "")),
                        "answer": answer,
                        "clue": self._extract_reason_clue(output, answer),
                        "message_type": info.get("message_type", "full"),
                    }
                )
                continue

            updated_info = dict(info)
            updated_info["output"] = output
            updated_info["message_type"] = "temporal_edge_unparseable"
            updated_info["source_message_type"] = info.get("message_type", "full")
            fallback_info[sender_id] = updated_info
            fallback_words += self._approx_word_count(output)

        if not parsed_entries:
            for sender_id, updated_info in fallback_info.items():
                self.num_compressed += 1
                self.original_words += self._approx_word_count(str(eligible_info[sender_id].get("output", "")))
                self.compressed_words += self._approx_word_count(str(updated_info.get("output", "")))
            return fallback_info

        vote_state, metadata = self._build_vote_state(
            parsed_entries,
            option_map or {},
            receiver_id,
            raw_inputs=raw_inputs,
            domain=domain,
            include_conflict_clues=include_conflict_clues,
        )
        compressed_info = {
            DEFAULT_TEMPORAL_VOTE_STATE_ID: {
                "role": "Temporal Vote State",
                "output": vote_state,
                "message_type": "temporal_vote_state",
                "source_message_type": "aggregate",
                "summary_mode": self.summary_mode,
                "senders": parsed_entries,
                "unparseable_senders": list(fallback_info.keys()),
                **metadata,
            }
        }
        compressed_info.update(fallback_info)

        self.num_compressed += len(parsed_entries) + len(fallback_info)
        self.original_words += original_words
        self.compressed_words += self._approx_word_count(vote_state) + fallback_words
        return compressed_info

    def _build_vote_state(
        self,
        parsed_entries: list[dict[str, str]],
        option_map: dict[str, str],
        receiver_id: str | None,
        raw_inputs: Any = None,
        domain: str = "",
        include_conflict_clues: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        self_entry = next((entry for entry in parsed_entries if entry["id"] == receiver_id), None)
        peer_entries = [
            entry for entry in parsed_entries if receiver_id is None or entry["id"] != receiver_id
        ]
        counted_entries = peer_entries if peer_entries else parsed_entries
        counts = Counter(entry["answer"] for entry in counted_entries)
        all_counts = Counter(entry["answer"] for entry in parsed_entries)
        total = len(counted_entries)
        labels = self._vote_labels(counts, option_map)
        count_text = ", ".join(f"{label}={counts.get(label, 0)}" for label in labels)
        max_count = max(counts.values())
        majority_labels = sorted(label for label, count in counts.items() if count == max_count)
        has_strict_majority = len(majority_labels) == 1 and max_count > total / 2
        has_reliable_majority = (
            len(majority_labels) == 1
            and max_count / total >= self.reliable_majority_threshold
        )
        is_consensus = len(majority_labels) == 1 and max_count == total
        is_conflict = not is_consensus and len(counts) > 1

        lines = [
            "Previous-round compressed temporal memory:",
            f"- Peer vote counts from received temporal edges: {count_text}.",
        ]
        voted_option_hints = [
            self._format_option_hint(label, option_map)
            for label in labels
            if counts.get(label, 0) > 0 and label in option_map
        ]
        if voted_option_hints:
            lines.append(f"- Voted option meanings: {'; '.join(voted_option_hints)}.")

        if is_consensus:
            consensus_label = majority_labels[0]
            lines.append(
                f"- All peers previously used label: {self._format_option_hint(consensus_label, option_map)} "
                f"({max_count}/{total})."
            )
        elif has_reliable_majority:
            majority_label = majority_labels[0]
            lines.append(
                f"- Most frequent peer label: {self._format_option_hint(majority_label, option_map)} "
                f"({max_count}/{total})."
            )
        else:
            leading = ", ".join(self._format_option_hint(label, option_map) for label in majority_labels)
            if has_strict_majority:
                lines.append(
                    f"- Divided vote; no reliable majority. Leading candidate: {leading} "
                    f"({max_count}/{total})."
                )
            else:
                lines.append(f"- No strict majority; leading candidates: {leading} ({max_count}/{total}).")

        if self_entry is not None:
            lines.append(f"- Your previous answer: {self._format_option_hint(self_entry['answer'], option_map)}.")

        individual_labels = "; ".join(
            f"{entry['id']}({entry['role'] or 'unknown role'})={entry['answer']}"
            for entry in parsed_entries
        )
        lines.append(f"- Individual previous labels: {individual_labels}.")
        if include_conflict_clues and is_conflict and self.reason_word_budget > 0:
            clue_lines = self._conflict_clue_lines(
                counted_entries=counted_entries,
                counts=counts,
                option_map=option_map,
            )
            if clue_lines:
                lines.append("- Compact rationale clues by candidate:")
                lines.extend(clue_lines)
        lines.append(
            "- Use this as compact peer memory, not proof. Re-check the question and options; "
            "if the vote conflicts with the option semantics, choose the option supported by the task."
        )
        if is_conflict:
            lines.append("- For divided votes, do not follow the largest count unless the task wording supports it.")
        guidance = self._task_guidance(raw_inputs, domain)
        if guidance:
            lines.append(guidance)
        metadata = {
            "vote_distribution": dict(counts),
            "all_vote_distribution": dict(all_counts),
            "majority_answer": majority_labels[0] if has_strict_majority else None,
            "reliable_majority_answer": majority_labels[0] if has_reliable_majority else None,
            "self_vote": self_entry["answer"] if self_entry is not None else None,
            "is_consensus": is_consensus,
            "is_conflict": is_conflict,
        }
        return "\n".join(lines), metadata

    def _conflict_clue_lines(
        self,
        counted_entries: list[dict[str, str]],
        counts: Counter,
        option_map: dict[str, str],
    ) -> list[str]:
        lines = []
        label_order = sorted(counts, key=lambda label: (-counts[label], label))[:3]
        for label in label_order:
            clue_entry = self._select_representative_clue(
                [entry for entry in counted_entries if entry["answer"] == label],
                option_map.get(label, ""),
            )
            if clue_entry is None:
                continue
            clue = self._truncate_words(clue_entry["clue"], self.reason_word_budget)
            option_hint = self._format_option_hint(label, option_map)
            lines.append(f"  * {option_hint} ({counts[label]} peer): {clue}")
        return lines

    def _select_representative_clue(
        self,
        entries: list[dict[str, str]],
        option_text: str,
    ) -> dict[str, str] | None:
        candidates = [entry for entry in entries if entry.get("clue", "").strip()]
        if not candidates:
            return None
        option_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9'-]+", option_text.lower()))
        if not option_terms:
            return min(candidates, key=lambda entry: len(entry["clue"]))

        def score(entry: dict[str, str]) -> tuple[int, int]:
            clue_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9'-]+", entry["clue"].lower()))
            overlap = len(option_terms & clue_terms)
            return overlap, -len(entry["clue"])

        return max(candidates, key=score)

    def _vote_labels(self, counts: Counter, option_map: dict[str, str]) -> list[str]:
        if option_map:
            return sorted(option_map.keys())
        return sorted(counts.keys())

    def _format_option_hint(self, label: str, option_map: dict[str, str]) -> str:
        option_text = option_map.get(label)
        if not option_text:
            return label
        return f"{label}: {self._truncate_words(option_text, self.option_word_budget)}"

    def _task_guidance(self, raw_inputs: Any, domain: str = "") -> str:
        if not isinstance(raw_inputs, dict):
            return ""
        task = str(raw_inputs.get("task", ""))
        guidance = [
            "First solve from the original task and options, then compare this memory."
        ]
        if re.search(
            r"\b(not|except|unless|false|least|incorrect|not usually true|not true)\b",
            task,
            flags=re.IGNORECASE,
        ):
            guidance.append(
                "The task has negative or exception wording; identify the requested polarity before choosing."
            )
        if domain in {"mmlu", "mmlu_pro"}:
            if "Scenario 1" in task and "Scenario 2" in task:
                guidance.append(
                    "For scenario morality questions, judge only stated facts; do not infer unstated permission, intent, or harm. "
                    "Map judgments exactly: A=Wrong/Wrong, B=Wrong/Not wrong, C=Not wrong/Wrong, D=Not wrong/Not wrong."
                )
            if re.search(
                r"\b(neglig|liable|admissible|self defense|damages|malpractice|causation)\b",
                task,
                flags=re.IGNORECASE,
            ):
                guidance.append(
                    "For law questions, cite the specific rule or exception in the option text before changing answers."
                )
        return "- " + " ".join(guidance)

    async def aaggregate(
        self,
        temporal_info: dict[str, dict],
        raw_inputs: Any = None,
        domain: str = "",
        receiver_id: str | None = None,
        receiver_role: str = "",
        receiver_description: str = "",
        receiver_agent: str = "",
    ) -> dict[str, dict]:
        return self.aggregate(
            temporal_info,
            raw_inputs=raw_inputs,
            domain=domain,
            receiver_id=receiver_id,
        )

    def summary(self) -> str:
        if self.num_compressed == 0:
            return "Temporal edge compressor compressed 0 messages."
        ratio = self.compressed_words / max(self.original_words, 1)
        return (
            f"Temporal {self.summary_mode} compressor compressed {self.num_compressed} messages "
            f"and filtered {self.num_role_filtered} low-reliability role messages "
            f"from {self.original_words} to {self.compressed_words} words "
            f"(keep ratio {ratio:.3f})."
        )

    def _eligible_temporal_info(self, temporal_info: dict[str, dict]) -> dict[str, dict]:
        eligible_info = {}
        for sender_id, info in temporal_info.items():
            output = info.get("output") if isinstance(info, dict) else None
            if not isinstance(output, str) or not output.strip():
                continue
            if "None." in output:
                continue
            role = str(info.get("role", ""))
            if role in self.ignored_sender_roles:
                self.num_role_filtered += 1
                continue
            eligible_info[sender_id] = info
        return eligible_info

    def _compress_output(self, output: str, option_map: dict[str, str] | None = None) -> str:
        answer = self._extract_answer_label(output)
        if answer:
            compressed = answer
            option_text = (option_map or {}).get(answer)
            if option_text and self.option_word_budget > 0:
                compressed = f"{answer}: {self._truncate_words(option_text, self.option_word_budget)}"
            reason_clue = self._extract_reason_clue(output, answer)
            if reason_clue and self.reason_word_budget > 0:
                compressed += f" | clue: {self._truncate_words(reason_clue, self.reason_word_budget)}"
            return compressed
        return output.strip()

    def _extract_reason_clue(self, output: str, answer: str) -> str:
        lines = output.splitlines()
        if not lines:
            return ""
        body = "\n".join(lines[1:]).strip()
        if not body:
            return ""

        body = re.sub(
            rf"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(?:option\s+)?{re.escape(answer)}\b\s*(?:because|as)?",
            "",
            body,
            count=1,
            flags=re.IGNORECASE,
        ).strip(" .:-\n\t")
        if not body:
            return ""

        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", body):
            sentence = sentence.strip(" -*\t")
            if sentence:
                return sentence
        return body

    def _extract_option_map(self, raw_inputs: Any) -> dict[str, str]:
        if not isinstance(raw_inputs, dict):
            return {}
        task = raw_inputs.get("task")
        if not isinstance(task, str):
            return {}

        option_map: dict[str, str] = {}
        option_pattern = re.compile(
            r"Option\s+([A-J]):\s*(.*?)(?=\nOption\s+[A-J]:|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in option_pattern.finditer(task):
            label = match.group(1).upper()
            option_text = " ".join(match.group(2).split())
            if option_text:
                option_map[label] = option_text
        return option_map

    def _truncate_words(self, text: str, budget: int) -> str:
        text = text.strip()
        if budget <= 0:
            return text
        words = re.findall(r"\S+", text)
        if len(words) <= budget:
            return text
        return " ".join(words[:budget]).rstrip(".,;:") + "..."

    def _extract_answer_label(self, output: str) -> str | None:
        if not output:
            return None

        lines = output.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(
                r"^(?:(?:answer\s*(?:is|:)?|final\s+answer\s*(?:is|:)?|option)\s*)?([A-J])(?:$|[).:：,-])",
                stripped,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).upper()
            break

        patterns = (
            r"\banswer\s+is\s+(?:option\s+)?([A-J])\b",
            r"\bfinal\s+answer\s*(?:is|:)\s*(?:option\s+)?([A-J])\b",
            r"\bcorrect\s+answer\s*(?:is|:)\s*(?:option\s+)?([A-J])\b",
            r"\boption\s+([A-J])\s+(?:is|seems|appears|remains)\b",
            r"\bchoose\s+(?:option\s+)?([A-J])\b",
            r"\b([A-J])\s+remains\s+(?:the\s+)?best\b",
        )
        for pattern in patterns:
            match = re.search(pattern, output, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _approx_word_count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


def add_temporal_api_summary_args(parser):
    parser.add_argument(
        "--temporal_api_summary",
        action="store_true",
        help="Compress previous-round temporal peer messages into compact answer-label memory.",
    )
    parser.add_argument(
        "--temporal_summary_mode",
        type=str,
        default="answer_only",
        choices=TEMPORAL_SUMMARY_MODES,
        help=(
            "Temporal memory compression mode. answer_only keeps compatibility; "
            "vote_state aggregates peer votes; conflict_clue adds short rationale clues only on divided votes."
        ),
    )
    parser.add_argument(
        "--temporal_option_word_budget",
        type=int,
        default=None,
        help="Optional word budget for option text in temporal summaries.",
    )
    parser.add_argument(
        "--temporal_reason_word_budget",
        type=int,
        default=None,
        help="Optional word budget for conflict rationale clues in temporal summaries.",
    )
    parser.add_argument(
        "--temporal_reliable_majority_threshold",
        type=float,
        default=DEFAULT_TEMPORAL_RELIABLE_MAJORITY_THRESHOLD,
        help="Peer-vote fraction required before temporal memory calls a majority reliable.",
    )
    parser.add_argument(
        "--temporal_full_history_edges",
        action="store_true",
        help=(
            "Allow temporal edges to read all previous-round nodes, including reverse and self history, "
            "instead of filtering them through the current spatial DAG cycle check."
        ),
    )


def temporal_api_summary_graph_kwargs(args) -> dict[str, Any]:
    return {
        "temporal_api_summary": getattr(args, "temporal_api_summary", False),
        "temporal_summary_mode": getattr(args, "temporal_summary_mode", "answer_only"),
        "temporal_option_word_budget": getattr(args, "temporal_option_word_budget", None),
        "temporal_reason_word_budget": getattr(args, "temporal_reason_word_budget", None),
        "temporal_reliable_majority_threshold": getattr(
            args,
            "temporal_reliable_majority_threshold",
            DEFAULT_TEMPORAL_RELIABLE_MAJORITY_THRESHOLD,
        ),
        "temporal_full_history_edges": getattr(args, "temporal_full_history_edges", False),
    }
