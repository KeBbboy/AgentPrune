import re
import string
import warnings
from typing import List, Optional


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(s: str, char_list: Optional[List[str]] = None) -> List[str]:
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    return normalize_str(model_answer) == normalize_str(ground_truth)


def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", str(input_str))
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()
