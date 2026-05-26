import asyncio
from functools import lru_cache
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import Time, Cost, PromptTokens, CompletionTokens


DEFAULT_MMLU_ROLES = [
    "Knowlegable Expert",
    "Critic",
    "Mathematician",
    "Doctor",
    "Lawyer",
    "Economist",
    "Programmer",
    "Historian",
    "Psychologist",
]

ROLE_SUBJECT_RELEVANCE_MODEL = "all-MiniLM-L6-v2"
ROLE_SUBJECT_RELEVANCE_REPO = f"sentence-transformers/{ROLE_SUBJECT_RELEVANCE_MODEL}"

ROLE_RELEVANCE_PROFILES = {
    "knowlegable expert": "general academic expert with broad interdisciplinary knowledge",
    "knowledgeable expert": "general academic expert with broad interdisciplinary knowledge",
    "critic": "critical reasoning expert evaluating arguments, assumptions, and evidence",
    "mathematician": "mathematics algebra statistics probability formal logic physics quantitative reasoning",
    "doctor": "medicine clinical knowledge anatomy biology nutrition virology healthcare diagnosis",
    "lawyer": "law jurisprudence legal reasoning politics public policy constitutional and professional law",
    "economist": "economics business accounting marketing management finance macroeconomics microeconomics",
    "programmer": "computer science programming software engineering machine learning electrical engineering",
    "historian": "history prehistory geography archaeology ancient cultures and historical evidence",
    "psychologist": "psychology sociology philosophy morality human behavior and moral scenarios",
}


@dataclass
class NeedAwarePullConfig:
    llm_name: str
    num_agents: int = 5
    confidence_threshold: float = 0.7
    entropy_threshold: float = 0.8
    max_requesters: int = 2
    helpers_per_request: int = 1
    max_response_words: int = 45
    fallback_entropy_threshold: float = 0.45
    evidence_weight: float = 0.35
    eval_batch_size: int = 4
    limit_questions: Optional[int] = None


@dataclass
class AgentState:
    agent_id: str
    role: str
    raw_output: str
    answer: str
    confidence: float
    uncertain_options: List[str]
    need: str
    rationale: str


@dataclass
class PullRequest:
    requester_id: str
    candidate_options: List[str]
    need: str
    trigger: str


@dataclass
class PullResponse:
    requester_id: str
    helper_id: str
    helper_role: str
    raw_output: str
    preferred_option: str
    evidence: str


class NeedAwarePullMMLU:
    """Round 0 independent answers, Round 1 pull communication, Round 2 aggregation."""

    def __init__(self, config: NeedAwarePullConfig):
        from AgentDropout.llm.llm_registry import LLMRegistry

        self.config = config
        self.llm = LLMRegistry.get(config.llm_name)
        self.roles = list(itertools.islice(itertools.cycle(DEFAULT_MMLU_ROLES), max(1, config.num_agents)))

    async def answer_record(self, record, dataset) -> Dict[str, Any]:
        input_dict = dataset.record_to_input(record)
        task = input_dict["task"]
        subject = input_dict.get("subject", "")

        states = await self._round0_independent_answers(task)
        requests = self._build_pull_requests(states)
        responses = await self._round1_pull_responses(task, subject, states, requests)
        final_answer = await self._round2_aggregate(task, subject, states, responses)

        return {
            "Question": task,
            "Subject": subject,
            "Response": final_answer,
            "Initial_states": [state.__dict__ for state in states],
            "Pull_requests": [request.__dict__ for request in requests],
            "Pull_responses": [response.__dict__ for response in responses],
        }

    async def _round0_independent_answers(self, task: str) -> List[AgentState]:
        tasks = []
        for idx, role in enumerate(self.roles):
            tasks.append(asyncio.create_task(self._independent_answer(f"agent_{idx}", role, task)))
        return list(await asyncio.gather(*tasks))

    async def _independent_answer(self, agent_id: str, role: str, task: str) -> AgentState:
        system_prompt = (
            f"You are a {role} solving a four-option MMLU multiple-choice question. "
            "Answer independently and expose uncertainty for communication planning."
        )
        user_prompt = (
            f"Question:\n{task}\n\n"
            "Return exactly these fields:\n"
            "Answer: one of A/B/C/D\n"
            "Confidence: a number from 0 to 1\n"
            "Uncertain options: comma-separated option letters, or None\n"
            "Need: what evidence would help if you are uncertain\n"
            "Rationale: one concise reason, under 50 words"
        )
        raw_output = await self._call_llm(system_prompt, user_prompt)
        parsed = parse_agent_state(raw_output)
        return AgentState(agent_id=agent_id, role=role, raw_output=raw_output, **parsed)

    def _build_pull_requests(self, states: List[AgentState]) -> List[PullRequest]:
        majority_answer = majority_vote([state.answer for state in states])
        vote_entropy = normalized_vote_entropy([state.answer for state in states])
        candidates: List[tuple[float, PullRequest]] = []

        for state in states:
            triggers = []
            if state.confidence < self.config.confidence_threshold:
                triggers.append("low_confidence")
            if state.answer and majority_answer and state.answer != majority_answer:
                triggers.append("minority_answer")
            if vote_entropy >= self.config.entropy_threshold and len(state.uncertain_options) >= 2:
                triggers.append("high_vote_entropy")
            if not triggers:
                continue

            options = state.uncertain_options[:]
            if len(options) < 2 and state.answer:
                options = [state.answer]
                if majority_answer and majority_answer not in options:
                    options.append(majority_answer)
            if len(options) < 2:
                options = ["A", "B"]

            need_score = (1 - state.confidence) + (0.3 if state.answer != majority_answer else 0.0)
            candidates.append(
                (
                    need_score,
                    PullRequest(
                        requester_id=state.agent_id,
                        candidate_options=options[:2],
                        need=state.need or f"Need evidence distinguishing {' vs '.join(options[:2])}.",
                        trigger=",".join(triggers),
                    ),
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [request for _, request in candidates[: self.config.max_requesters]]

    async def _round1_pull_responses(
        self,
        task: str,
        subject: str,
        states: List[AgentState],
        requests: List[PullRequest],
    ) -> List[PullResponse]:
        tasks = []
        for request in requests:
            for helper in self._select_helpers(subject, states, request):
                tasks.append(asyncio.create_task(self._answer_pull_request(task, subject, states, request, helper)))
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))

    def _select_helpers(self, subject: str, states: List[AgentState], request: PullRequest) -> List[AgentState]:
        requester = next(state for state in states if state.agent_id == request.requester_id)
        scored_helpers = []
        for helper in states:
            if helper.agent_id == request.requester_id:
                continue
            disagreement = 1.0 if helper.answer and helper.answer != requester.answer else 0.0
            option_relevance = 1.0 if helper.answer in request.candidate_options else 0.0
            score = (
                0.4 * helper.confidence
                + 0.3 * role_subject_relevance(helper.role, subject)
                + 0.2 * disagreement
                + 0.1 * option_relevance
            )
            scored_helpers.append((score, helper))
        scored_helpers.sort(key=lambda item: item[0], reverse=True)
        return [helper for _, helper in scored_helpers[: self.config.helpers_per_request]]

    async def _answer_pull_request(
        self,
        task: str,
        subject: str,
        states: List[AgentState],
        request: PullRequest,
        helper: AgentState,
    ) -> PullResponse:
        requester = next(state for state in states if state.agent_id == request.requester_id)
        option_text = " vs ".join(request.candidate_options)
        system_prompt = (
            f"You are a {helper.role}. Provide only local contrastive evidence for a pull request. "
            "Do not solve the whole task again."
        )
        user_prompt = (
            f"Subject: {subject}\n"
            f"Question:\n{task}\n\n"
            f"Requester answer: {requester.answer}\n"
            f"Requester need: {request.need}\n"
            f"Requester is uncertain between: {option_text}\n"
            f"Your initial answer: {helper.answer}\n"
            f"Your initial rationale: {helper.rationale}\n\n"
            f"Give one piece of evidence distinguishing {option_text}. "
            f"Keep it under {self.config.max_response_words} words.\n"
            "Return exactly:\n"
            "Preferred option: A/B/C/D\n"
            "Evidence: one concise contrastive reason"
        )
        raw_output = await self._call_llm(system_prompt, user_prompt)
        preferred_option, evidence = parse_pull_response(raw_output)
        return PullResponse(
            requester_id=request.requester_id,
            helper_id=helper.agent_id,
            helper_role=helper.role,
            raw_output=raw_output,
            preferred_option=preferred_option,
            evidence=evidence,
        )

    async def _round2_aggregate(
        self,
        task: str,
        subject: str,
        states: List[AgentState],
        responses: List[PullResponse],
    ) -> str:
        return conservative_weighted_vote(
            states=states,
            responses=responses,
            subject=subject,
            fallback_entropy_threshold=self.config.fallback_entropy_threshold,
            evidence_weight=self.config.evidence_weight,
        )

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.llm.agen(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if isinstance(response, list):
            return str(response[0]) if response else ""
        return str(response)


async def evaluate_need_aware_pull(dataset, args) -> float:
    config = NeedAwarePullConfig(
        llm_name=args.llm_name,
        num_agents=sum(args.agent_nums),
        confidence_threshold=args.pull_confidence_threshold,
        entropy_threshold=args.pull_entropy_threshold,
        max_requesters=args.pull_max_requesters,
        helpers_per_request=args.pull_helpers_per_request,
        max_response_words=args.pull_max_response_words,
        fallback_entropy_threshold=args.pull_fallback_entropy_threshold,
        evidence_weight=args.pull_evidence_weight,
        eval_batch_size=args.batch_size,
        limit_questions=args.pull_limit_questions,
    )
    runner = NeedAwarePullMMLU(config)
    total = 0
    correct = 0
    records = []

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/{getattr(args, 'domain', 'mmlu')}")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_napc_{current_time}.json"

    for batch in tqdm(list(batch_records(dataset, config.eval_batch_size, config.limit_questions))):
        results = await asyncio.gather(*[asyncio.create_task(runner.answer_record(record, dataset)) for record in batch])
        for result, record in zip(results, batch):
            answer = dataset.postprocess_answer(result["Response"])
            target = dataset.record_to_target_answer(record)
            total += 1
            correct += int(answer == target)
            result["Answer"] = target
            result["Postprocessed_answer"] = answer
            result["Solved"] = answer == target
            result["Accuracy"] = correct / total
            records.append(result)
            print(f"NAPC answer={answer} target={target} accuracy={correct}/{total}")
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=4)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    accuracy = correct / total if total else 0.0
    print(f"NAPC Score: {accuracy}")
    return accuracy


def batch_records(dataset, batch_size: int, limit_questions: Optional[int]) -> Iterable[List[Any]]:
    batch = []
    max_items = len(dataset) if limit_questions is None else min(len(dataset), limit_questions)
    for idx in range(max_items):
        batch.append(dataset[idx])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_agent_state(text: str) -> Dict[str, Any]:
    return {
        "answer": parse_option(field_value(text, "Answer") or text),
        "confidence": parse_confidence(field_value(text, "Confidence")),
        "uncertain_options": parse_options(field_value(text, "Uncertain options")),
        "need": field_value(text, "Need"),
        "rationale": field_value(text, "Rationale") or compact_text(text, 250),
    }


def parse_pull_response(text: str) -> tuple[str, str]:
    preferred = parse_option(field_value(text, "Preferred option") or text)
    evidence = field_value(text, "Evidence") or compact_text(text, 200)
    return preferred, evidence


def field_value(text: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_option(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else text.strip()[:1].upper()


def parse_options(text: str) -> List[str]:
    if not text or text.strip().lower() == "none":
        return []
    options = []
    for match in re.findall(r"\b([ABCD])\b", text, flags=re.IGNORECASE):
        option = match.upper()
        if option not in options:
            options.append(option)
    return options


def parse_confidence(text: str) -> float:
    if not text:
        return 0.5
    match = re.search(r"0?\.\d+|1(?:\.0+)?|0(?:\.0+)?|\d{1,3}\s*%", text)
    if not match:
        return 0.5
    raw = match.group(0).replace("%", "").strip()
    value = float(raw)
    if "%" in match.group(0) or value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def normalized_vote_entropy(answers: List[str]) -> float:
    counts = {option: answers.count(option) for option in ("A", "B", "C", "D")}
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            probability = count / total
            entropy -= probability * math.log(probability)
    return entropy / math.log(4)


def majority_vote(answers: List[str]) -> str:
    counts = {option: answers.count(option) for option in ("A", "B", "C", "D")}
    return max(counts.items(), key=lambda item: item[1])[0]


def conservative_weighted_vote(
    states: List[AgentState],
    responses: List[PullResponse],
    subject: str,
    fallback_entropy_threshold: float = 0.45,
    evidence_weight: float = 0.35,
) -> str:
    answers = [state.answer for state in states if state.answer in {"A", "B", "C", "D"}]
    fallback_answer = majority_vote(answers)
    if normalized_vote_entropy(answers) <= fallback_entropy_threshold or not responses:
        return fallback_answer

    scores = {option: 0.0 for option in ("A", "B", "C", "D")}
    for state in states:
        if state.answer not in scores:
            continue
        role_bonus = 0.25 * role_subject_relevance(state.role, subject)
        scores[state.answer] += max(0.05, state.confidence) * (1.0 + role_bonus)

    for response in responses:
        if response.preferred_option not in scores:
            continue
        helper_relevance = role_subject_relevance(response.helper_role, subject)
        scores[response.preferred_option] += evidence_weight * (1.0 + helper_relevance)

    weighted_answer = max(scores.items(), key=lambda item: item[1])[0]
    if scores[weighted_answer] <= scores[fallback_answer]:
        return fallback_answer
    return weighted_answer


@lru_cache(maxsize=1)
def _role_subject_embedding_model():
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROLE_SUBJECT_RELEVANCE_REPO)
    model = AutoModel.from_pretrained(ROLE_SUBJECT_RELEVANCE_REPO)
    model.eval()
    return tokenizer, model


def _normalize_relevance_text(text: str) -> str:
    return " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())


def _role_profile(role: str) -> str:
    normalized_role = _normalize_relevance_text(role)
    return ROLE_RELEVANCE_PROFILES.get(normalized_role, f"{normalized_role} academic expert")


@lru_cache(maxsize=1024)
def _embedding_cosine_similarity(left: str, right: str) -> float:
    import torch
    import torch.nn.functional as F

    tokenizer, model = _role_subject_embedding_model()
    encoded = tokenizer([left, right], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)

    token_embeddings = output.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
    embeddings = F.normalize(pooled, p=2, dim=1)
    return float((embeddings[0] * embeddings[1]).sum().item())


@lru_cache(maxsize=1024)
def role_subject_relevance(role: str, subject: str) -> float:
    role_text = _role_profile(role)
    subject_text = _normalize_relevance_text(subject)
    if not role_text.strip() or not subject_text:
        return 0.0

    similarity = _embedding_cosine_similarity(role_text, subject_text)
    return max(0.0, min(1.0, similarity))


def compact_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
