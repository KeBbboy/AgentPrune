import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentDropout_exp import need_aware_pull_mmlu as napc
from AgentDropout_exp.need_aware_pull_mmlu import (
    AgentState,
    NeedAwarePullConfig,
    NeedAwarePullMMLU,
    ROLE_SUBJECT_RELEVANCE_MODEL,
    conservative_weighted_vote,
    normalized_vote_entropy,
    parse_agent_state,
    parse_pull_response,
    role_subject_relevance,
)


def make_runner():
    runner = object.__new__(NeedAwarePullMMLU)
    runner.config = NeedAwarePullConfig(llm_name="mock", max_requesters=2, helpers_per_request=1)
    return runner


def test_parse_agent_state_extracts_uncertainty_fields():
    state = parse_agent_state(
        "Answer: C\n"
        "Confidence: 0.62\n"
        "Uncertain options: A, C\n"
        "Need: evidence distinguishing A from C\n"
        "Rationale: C fits the definition better."
    )

    assert state["answer"] == "C"
    assert state["confidence"] == 0.62
    assert state["uncertain_options"] == ["A", "C"]
    assert "distinguishing" in state["need"]


def test_parse_pull_response_extracts_preferred_option_and_evidence():
    preferred, evidence = parse_pull_response(
        "Preferred option: B\n"
        "Evidence: B names the clinical symptom, while D names a treatment."
    )

    assert preferred == "B"
    assert evidence.startswith("B names")


def test_build_pull_requests_prioritizes_low_confidence_minority_agent():
    runner = make_runner()
    states = [
        AgentState("agent_0", "Doctor", "", "A", 0.9, [], "", ""),
        AgentState("agent_1", "Critic", "", "A", 0.8, [], "", ""),
        AgentState("agent_2", "Lawyer", "", "C", 0.4, ["A", "C"], "Need contrast", ""),
    ]

    requests = runner._build_pull_requests(states)

    assert len(requests) == 1
    assert requests[0].requester_id == "agent_2"
    assert requests[0].candidate_options == ["A", "C"]


def test_helper_selection_uses_subject_role_relevance(monkeypatch):
    def fake_role_subject_relevance(role, subject):
        return 1.0 if role == "Doctor" and subject == "clinical_knowledge" else 0.0

    monkeypatch.setattr(napc, "role_subject_relevance", fake_role_subject_relevance)
    runner = make_runner()
    states = [
        AgentState("agent_0", "Lawyer", "", "A", 0.9, [], "", ""),
        AgentState("agent_1", "Doctor", "", "B", 0.8, [], "", ""),
        AgentState("agent_2", "Critic", "", "A", 0.4, ["A", "B"], "Need contrast", ""),
    ]
    request = runner._build_pull_requests(states)[0]

    helpers = runner._select_helpers("clinical_knowledge", states, request)

    assert helpers[0].role == "Doctor"


def test_entropy_and_role_relevance_helpers():
    assert normalized_vote_entropy(["A", "A", "A", "A"]) == 0.0
    assert normalized_vote_entropy(["A", "B", "C", "D"]) == 1.0


def test_role_subject_relevance_uses_minilm_model(monkeypatch):
    assert ROLE_SUBJECT_RELEVANCE_MODEL == "all-MiniLM-L6-v2"

    role_subject_relevance.cache_clear()
    monkeypatch.setattr(napc, "_embedding_cosine_similarity", lambda left, right: 0.42)

    assert role_subject_relevance("Doctor", "clinical_knowledge") == 0.42
    role_subject_relevance.cache_clear()


def test_conservative_vote_falls_back_when_consensus_is_strong():
    states = [
        AgentState("agent_0", "Doctor", "", "A", 0.8, [], "", ""),
        AgentState("agent_1", "Lawyer", "", "A", 0.7, [], "", ""),
        AgentState("agent_2", "Critic", "", "A", 0.7, [], "", ""),
    ]

    assert conservative_weighted_vote(states, [], "clinical_knowledge") == "A"


def test_conservative_vote_uses_evidence_only_for_high_entropy_cases(monkeypatch):
    monkeypatch.setattr(napc, "role_subject_relevance", lambda role, subject: 1.0 if role == "Doctor" else 0.0)
    states = [
        AgentState("agent_0", "Doctor", "", "A", 0.6, [], "", ""),
        AgentState("agent_1", "Lawyer", "", "B", 0.6, [], "", ""),
        AgentState("agent_2", "Critic", "", "C", 0.6, [], "", ""),
        AgentState("agent_3", "Economist", "", "D", 0.6, [], "", ""),
    ]
    response = type(
        "Response",
        (),
        {"preferred_option": "A", "helper_role": "Doctor"},
    )()

    assert conservative_weighted_vote(states, [response], "clinical_knowledge") == "A"
