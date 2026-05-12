import json
import re
import string
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence


LONG_BENCH_FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}
LONG_BENCH_ZH_QA_DATASETS = {"multifieldqa_zh"}
LONG_BENCH_ZH_SUMMARY_DATASETS = {"dureader", "vcsum"}
LONG_BENCH_SUMMARY_DATASETS = {"gov_report", "qmsum", "multi_news", "samsum"}
LONG_BENCH_CLASSIFICATION_DATASETS = {"trec", "lsht"}
LONG_BENCH_RETRIEVAL_DATASETS = {"passage_retrieval_en"}
LONG_BENCH_ZH_RETRIEVAL_DATASETS = {"passage_retrieval_zh"}
LONG_BENCH_COUNT_DATASETS = {"passage_count"}
LONG_BENCH_CODE_DATASETS = {"lcc", "repobench-p"}

OFFICIAL_LONG_BENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        'information in the article, write "unanswerable". If the question is a yes/no question, answer '
        '"yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        'write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
        "Do not provide any explanation. "
        "Question: {input}\n\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do "
        "not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n"
        "{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n"
        "文章：{context}\n\n"
        "请基于上述文章回答下面的问题。\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. Answer "
        "the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n"
        "会议记录：\n{context}\n\n"
        "会议总结："
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n"
        "{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "lsht": (
        "请判断给定新闻的类别，下面是一些例子。\n\n"
        "{context}\n"
        "{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which "
        "paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n"
        "{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must "
        "be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n"
        "{context}\n\n"
        "下面是一个摘要\n\n"
        "{input}\n\n"
        "请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n"
        "答案是："
    ),
    "lcc": (
        "Please complete the code given below. \n"
        "{context}"
        "Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n"
        "{context}"
        "{input}"
        "Next line of code:\n"
    ),
}

QASPER_REASONING_PROMPT = (
    "You are given a scientific article and a question. Answer the question as concisely as you can, "
    "using a single phrase or sentence if possible. If the question cannot be answered based on the "
    'information in the article, write "unanswerable". If the question is a yes/no question, answer '
    '"yes", "no", or "unanswerable". First answer the question, then briefly explain your reasoning.\n\n'
    "Article: {context}\n\n"
    "Answer the question based on the above article as concisely as you can, using a single phrase or "
    "sentence if possible. If the question cannot be answered based on the information in the article, "
    'write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
    "First answer the question, then briefly explain your reasoning. "
    "Question: {input}\n\n"
    "Answer:"
)


def _read_records(dataset_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"LongBench dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "items", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported file format for LongBench dataset: {path.suffix}")


def resolve_longbench_path(dataset_path: str | Path, dataset_name: Optional[str] = None) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path
    if dataset_name is None:
        raise ValueError("--dataset_name is required when --dataset_path points to a directory")

    candidates = [
        path / f"{dataset_name}.jsonl",
        path / f"{dataset_name}.json",
        path / "data" / f"{dataset_name}.jsonl",
        path / "data" / f"{dataset_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No LongBench file found for '{dataset_name}'. Tried: {tried}")


def _first_non_empty(record: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return default


def _extract_answers(record: Dict[str, Any]) -> List[str]:
    for key in ("answers", "answer", "target", "targets", "label", "labels"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if str(value).strip():
            return [str(value).strip()]
    raise KeyError(f"Unable to find an answer field in record: {record}")


def _extract_all_classes(record: Dict[str, Any]) -> List[str]:
    value = record.get("all_classes") or record.get("classes") or record.get("choices")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _format_task(record: Dict[str, Any], dataset_name: str, with_reasoning: bool = False) -> str:
    context = _first_non_empty(record, ("context", "passage", "document", "documents", "article"))
    user_input = _first_non_empty(record, ("input", "question", "query", "prompt", "instruction"))
    all_classes = _extract_all_classes(record)
    dataset_key = _base_dataset_name(dataset_name)

    if dataset_key == "qasper" and with_reasoning:
        prompt_template = QASPER_REASONING_PROMPT
    else:
        prompt_template = OFFICIAL_LONG_BENCH_PROMPTS.get(dataset_key)
    if prompt_template is not None:
        return prompt_template.format(context=context, input=user_input)

    parts = [f"Dataset: {dataset_name}"]
    if all_classes:
        parts.append("Candidate answers:\n" + "\n".join(f"- {item}" for item in all_classes))
    if user_input:
        parts.append(f"Question/Input:\n{user_input}")
    if context:
        parts.append(f"Context:\n{context}")
    if not user_input and not context:
        raise KeyError(f"Unable to find input/context fields in record: {record}")
    return "\n\n".join(parts)


def load_longbench_dataset(
    dataset_path: str | Path = "dataset/longbench",
    dataset_name: Optional[str] = None,
    with_reasoning: bool = False,
) -> List[Dict[str, Any]]:
    resolved_path = resolve_longbench_path(dataset_path, dataset_name)
    inferred_dataset_name = dataset_name or resolved_path.stem
    processed_records: List[Dict[str, Any]] = []

    for index, raw_record in enumerate(_read_records(resolved_path)):
        processed_records.append(
            {
                "id": raw_record.get("_id") or raw_record.get("id") or index,
                "dataset": inferred_dataset_name,
                "task": _format_task(raw_record, inferred_dataset_name, with_reasoning=with_reasoning),
                "input": _first_non_empty(raw_record, ("input", "question", "query", "prompt", "instruction")),
                "context": _first_non_empty(raw_record, ("context", "passage", "document", "documents", "article")),
                "answers": _extract_answers(raw_record),
                "all_classes": _extract_all_classes(raw_record),
                "length": raw_record.get("length"),
                "raw": raw_record,
            }
        )
    return processed_records


def extract_answer_text(answer: Any, first_line_only: bool = False) -> str:
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = answer.split("\n\n", 1)[0].strip()
    answer = re.split(r"\n\s*Reasoning\s*[:：]", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    answer = re.sub(r"^\s*Answer\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
    if first_line_only:
        answer_line_match = re.search(
            r"^\s*(?:answer\s*[:：]\s*)?(.+?)\s*$",
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if answer_line_match is not None:
            return answer_line_match.group(1).strip()
    return answer


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _normalize_zh_answer(text: str) -> str:
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
        "｟｠｢｣､、〃》「」『』〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿–—"
        "‘’‛“”„‟…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    text = text.lower()
    text = "".join(char for char in text if char not in all_punctuation)
    return "".join(text.split())


def _normalize_answer(text: str) -> str:
    text = extract_answer_text(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    exclude = set(string.punctuation)
    text = "".join(" " if char in exclude else char for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_answer(text)
    if not normalized:
        return []
    if _contains_cjk(normalized):
        return [char for char in normalized if not char.isspace()]
    return normalized.split()


def _f1_from_tokens(prediction_tokens: Sequence[str], ground_truth_tokens: Sequence[str]) -> float:
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    common_count = sum(common.values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _qa_f1_score(prediction: str, ground_truth: str) -> float:
    return _f1_from_tokens(_normalize_answer(prediction).split(), _normalize_answer(ground_truth).split())


def _zh_tokens(text: str) -> List[str]:
    try:
        import jieba

        raw_tokens = list(jieba.cut(text, cut_all=False))
    except Exception:
        raw_tokens = list(text)
    return [_normalize_zh_answer(token) for token in raw_tokens if _normalize_zh_answer(token)]


def _qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    return _f1_from_tokens(_zh_tokens(prediction), _zh_tokens(ground_truth))


def _rouge_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _tokenize(prediction)
    ground_truth_tokens = _tokenize(ground_truth)
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    lcs = _lcs_len(prediction_tokens, ground_truth_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_zh_score(prediction: str, ground_truth: str) -> float:
    return _rouge_score(" ".join(_zh_tokens(prediction)), " ".join(_zh_tokens(ground_truth)))


def _classification_score(prediction: str, ground_truth: str, all_classes: List[str]) -> float:
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)

    for match_term in em_match_list.copy():
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)

    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def _count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if len(numbers) == 0:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(right_num / len(numbers))


def _retrieval_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if len(numbers) == 0:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(right_num / len(numbers))


def _retrieval_zh_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if len(numbers) == 0:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(right_num / len(numbers))


def _code_sim_score(prediction: str, ground_truth: str) -> float:
    selected_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            selected_line = line
            break
    return SequenceMatcher(None, selected_line, ground_truth).ratio()


def _base_dataset_name(dataset_name: str) -> str:
    return dataset_name[:-2] if dataset_name.endswith("_e") else dataset_name


def _preprocess_prediction_for_dataset(prediction: str, dataset_name: str) -> str:
    if _base_dataset_name(dataset_name) in LONG_BENCH_FIRST_LINE_DATASETS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def _lcs_len(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for j, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def score_longbench_answer(
    prediction: Any,
    answers: List[str],
    dataset_name: str,
    all_classes: Optional[List[str]] = None,
) -> float:
    prediction_text = extract_answer_text(prediction)
    all_classes = all_classes or []
    dataset_key = _base_dataset_name(dataset_name.lower())
    prediction_text = _preprocess_prediction_for_dataset(prediction_text, dataset_key)

    if dataset_key in LONG_BENCH_ZH_QA_DATASETS:
        return max(_qa_f1_zh_score(prediction_text, answer) for answer in answers)
    if dataset_key in LONG_BENCH_ZH_SUMMARY_DATASETS:
        return max(_rouge_zh_score(prediction_text, answer) for answer in answers)
    if dataset_key in LONG_BENCH_SUMMARY_DATASETS:
        return max(_rouge_score(prediction_text, answer) for answer in answers)
    if dataset_key in LONG_BENCH_CLASSIFICATION_DATASETS:
        return max(_classification_score(prediction_text, answer, all_classes) for answer in answers)
    if dataset_key in LONG_BENCH_RETRIEVAL_DATASETS:
        return max(_retrieval_score(prediction_text, answer) for answer in answers)
    if dataset_key in LONG_BENCH_ZH_RETRIEVAL_DATASETS:
        return max(_retrieval_zh_score(prediction_text, answer) for answer in answers)
    if dataset_key in LONG_BENCH_COUNT_DATASETS:
        return max(_count_score(prediction_text, answer) for answer in answers)
    if dataset_key in LONG_BENCH_CODE_DATASETS:
        return max(_code_sim_score(prediction_text, answer) for answer in answers)
    return max(_qa_f1_score(prediction_text, answer) for answer in answers)
