import re
import string
from collections import Counter
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from AgentDropout.benchmarks.evaluators.gaia_scorer import question_scorer

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = strip_latex_wrappers(answer)
    answer = answer.lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"\btrue\b", "yes", answer)
    answer = re.sub(r"\bfalse\b", "no", answer)
    return " ".join(answer.split()).strip()


def strip_latex_wrappers(ans: str) -> str:
    if not ans:
        return ans

    s = str(ans).strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()

    for cmd in ("text", "boxed"):
        match = re.fullmatch(rf"\s*\\{cmd}\{{(.+)\}}\s*", s)
        if match:
            s = match.group(1).strip()
            break
        prefix = f"\\{cmd}" + "{"
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    return s


def _try_parse_date(s: str) -> Optional[date]:
    try:
        from dateutil import parser as date_parser

        dt = date_parser.parse(s, default=datetime(1, 1, 1), ignoretz=True)
        if dt.year == 1:
            return None
        return dt.date()
    except Exception:
        return None


def exact_match(prediction: str, ground_truth: str, case_sensitive: bool = False) -> bool:
    if not case_sensitive:
        prediction, ground_truth = prediction.lower(), ground_truth.lower()
    return prediction.strip() == ground_truth.strip()


def normalized_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if gt_norm in pred_norm:
        return True
    gt_no_initials = " ".join(t for t in gt_norm.split() if len(t) > 1)
    pred_no_initials = " ".join(t for t in pred_norm.split() if len(t) > 1)
    return bool(gt_no_initials) and gt_no_initials in pred_no_initials


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


MATH_TOKEN_PATTERN = re.compile(r"^\s*[\s+\-*/^%\d.()\{\}a-zA-Z\\]+\s*$")


def is_math_answer(s: str) -> bool:
    s = str(s).strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass

    if len(s) >= 2 and re.fullmatch(r"[A-Za-z]+", s) is not None:
        return False

    return bool(MATH_TOKEN_PATTERN.match(s))


def _normalize_latex_for_parse(s: str) -> str:
    return re.sub(r"\\[dct]frac\b", r"\\frac", str(s))


def evaluate_with_math_verify(prediction: str, ground_truth: str) -> bool:
    if parse is not None and verify is not None:
        try:
            gold = parse(_normalize_latex_for_parse(ground_truth))
            answer = parse(_normalize_latex_for_parse(prediction))
            if gold and answer:
                return bool(verify(gold, answer))
        except Exception:
            pass
    return normalized_match(prediction, ground_truth)


def _evaluate_answer_single(
    prediction: str,
    ground_truth: str,
    choices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    pred = (prediction or "").strip()
    gt = (ground_truth or "").strip()

    stripped_pred = strip_latex_wrappers(pred)
    stripped_gt = strip_latex_wrappers(gt)
    if normalize_answer(stripped_pred) == normalize_answer(stripped_gt):
        return {"correct": True, "accuracy": 1.0, "em": 1.0, "f1": 1.0}

    if choices is not None:
        score = float(_mc_correct(pred, gt, choices))
        em_score = score
        f1_score = score
    elif is_math_answer(gt):
        score = float(evaluate_with_math_verify(pred, gt))
        em_score = float(exact_match(pred, gt, case_sensitive=False))
        f1_score = token_f1(pred, gt)
    else:
        pred_date = _try_parse_date(pred)
        gt_date = _try_parse_date(gt)
        if pred_date is not None and gt_date is not None and pred_date == gt_date:
            return {"correct": True, "accuracy": 1.0, "em": 1.0, "f1": 1.0}

        score = float(question_scorer(pred, gt))
        em_score = float(exact_match(pred, gt, case_sensitive=False))
        f1_score = token_f1(pred, gt)

    if choices is None and contains_match(pred, gt):
        score = max(score, 1.0)
        em_score = max(em_score, 1.0)
        f1_score = max(f1_score, token_f1(pred, gt))

    return {"correct": score > 0, "accuracy": score, "em": em_score, "f1": f1_score}


def evaluate_answer(
    prediction: str,
    ground_truth: str,
    choices: Optional[List[str]] = None,
    answer_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    result = _evaluate_answer_single(prediction, ground_truth, choices)
    if not answer_aliases or result["accuracy"] >= 1.0:
        return result
    for alias in answer_aliases:
        if alias == ground_truth:
            continue
        alias_result = _evaluate_answer_single(prediction, alias, choices)
        if alias_result["accuracy"] > result["accuracy"]:
            result = alias_result
        if result["accuracy"] >= 1.0:
            break
    return result


def _mc_correct(prediction: str, ground_truth: str, choices: List[str]) -> bool:
    gt_norm = ground_truth.strip().upper()
    pred_norm = prediction.strip().upper()

    pred_letter: Optional[str] = None
    if len(pred_norm) == 1 and pred_norm in "ABCDE":
        pred_letter = pred_norm
    else:
        match = re.search(r"\b([A-E])\b", pred_norm)
        if match:
            pred_letter = match.group(1)

    if pred_letter == gt_norm:
        return True

    if choices:
        pred_text_norm = normalize_answer(prediction)
        for i, choice in enumerate(choices):
            if normalize_answer(choice) == pred_text_norm:
                return chr(ord("A") + i) == gt_norm

    return False


def evaluate_gaia(prediction: str, ground_truth: str) -> Dict[str, Any]:
    pred = (prediction or "").strip()
    gt = (ground_truth or "").strip()
    score = float(question_scorer(pred, gt))
    em_score = float(exact_match(pred, gt, case_sensitive=False))
    f1_score = token_f1(pred, gt)
    return {"correct": score > 0, "accuracy": score, "em": em_score, "f1": f1_score}


def evaluate_math(prediction: str, ground_truth: str) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth)


def evaluate_gpqa(prediction: str, ground_truth: str, choices: List[str]) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth, choices=choices)


def evaluate_qa(prediction: str, ground_truth: str) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth)


def evaluate_musique(
    prediction: str,
    ground_truth: str,
    answer_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    base_result = evaluate_answer(prediction, ground_truth)
    if base_result.get("correct", False):
        return base_result

    for alias in answer_aliases or []:
        if not alias:
            continue
        alias_result = evaluate_answer(prediction, alias)
        if alias_result.get("correct", False):
            return alias_result
    return base_result
