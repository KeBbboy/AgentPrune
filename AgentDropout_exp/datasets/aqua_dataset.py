import re

from dataset.aqua_dataset import *  # noqa: F401,F403


def aqua_get_predict(pred_str):
    if isinstance(pred_str, list):
        pred_str = pred_str[0] if pred_str else ""
    if not isinstance(pred_str, str):
        pred_str = str(pred_str)

    xml_match = re.search(r"<answer>\s*([ABCDE])\s*</answer>", pred_str, flags=re.IGNORECASE)
    if xml_match is not None:
        return xml_match.group(1).upper()

    answer_match = re.search(r"answer is\s*[:\-]?\s*([ABCDE])", pred_str, flags=re.IGNORECASE)
    if answer_match is not None:
        return answer_match.group(1).upper()

    option_match = re.search(r"\b([ABCDE])\b", pred_str, flags=re.IGNORECASE)
    if option_match is not None:
        return option_match.group(1).upper()

    return pred_str[:1].upper() if pred_str else ""


def aqua_data_process(dataset):
    records = []
    for data in dataset:
        options = data.get("options") or data.get("choices") or []
        if isinstance(options, dict):
            option_values = [options.get(letter) or options.get(letter.lower()) or "" for letter in "ABCDE"]
        else:
            option_values = list(options)
        option_values = [str(value) for value in option_values[:5]]
        while len(option_values) < 5:
            option_values.append("")

        task = (
            f"{data.get('question', data.get('Question', ''))}\n"
            f"Option A: {option_values[0]}\n"
            f"Option B: {option_values[1]}\n"
            f"Option C: {option_values[2]}\n"
            f"Option D: {option_values[3]}\n"
            f"Option E: {option_values[4]}"
        )
        answer = data.get("correct") or data.get("correct_option") or data.get("answer") or data.get("Answer", "")
        records.append({"task": task, "answer": str(answer).strip().upper()[:1]})
    return records
