import re
import random
from pathlib import Path
from typing import Dict, Iterator, List

from AgentPrune.tools.reader.readers import JSONLReader

def gsm_data_process(dataset):
    # extract the question, step and answer
    list_data_dict = []
    for data in dataset:
        item = {"task":data["question"]}
        raw_answer = data["answer"]
        raw_answer_list = raw_answer.split("\n####")
        item["step"] = raw_answer_list[0].strip()
        item["answer"] = raw_answer_list[-1].replace(",", "").strip()
        list_data_dict.append(item)

    return list_data_dict


def gsm_get_predict(pred_str):
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else: pred = ''
    
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
            
    pred=_strip_string(pred)
    
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
        
    if pred.isdigit():
        return pred
    else:
        matches = re.findall(r'\d+', pred)
        return matches[-1] if matches else '0'


class GSM8KDataset:
    def __init__(
        self,
        split: str,
        dataset_dir: str = "dataset/gsm8k",
        dataset_path: str | None = "dataset/gsm8k/gsm8k.jsonl",
        shuffle_seed: int = 888,
    ) -> None:
        self._split = split
        self._dataset_dir = Path(dataset_dir)
        self._dataset_path = Path(dataset_path) if dataset_path is not None else None
        self._shuffle_seed = shuffle_seed
        self._records = self._load_data()

    @staticmethod
    def get_domain() -> str:
        return "gsm8k"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self._records[index]

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self._records)

    def _candidate_paths(self) -> List[Path]:
        split_aliases = {
            "train": ["train", "dev"],
            "dev": ["dev", "train"],
            "val": ["val", "validation", "test"],
            "validation": ["validation", "val", "test"],
            "test": ["test", "val", "validation"],
        }
        split_names = split_aliases.get(self._split, [self._split])
        candidates: List[Path] = []
        for split_name in split_names:
            candidates.append(self._dataset_dir / f"{split_name}.jsonl")
            candidates.append(self._dataset_dir / f"gsm8k_{split_name}.jsonl")
        return candidates

    def _resolve_dataset_path(self) -> Path:
        for path in self._candidate_paths():
            if path.exists():
                return path
        if self._dataset_path is not None and self._dataset_path.exists():
            print(
                f"No split-specific file found for '{self._split}'. "
                f"Falling back to {self._dataset_path}."
            )
            return self._dataset_path
        raise FileNotFoundError(
            f"No dataset split file found for '{self._split}'. Tried: "
            + ", ".join(str(path) for path in self._candidate_paths())
        )

    def _load_data(self) -> List[Dict[str, str]]:
        dataset = JSONLReader.parse_file(self._resolve_dataset_path())
        records = gsm_data_process(dataset)
        random.Random(self._shuffle_seed).shuffle(records)
        print("Total number of questions: ", len(records))
        return records

    @staticmethod
    def record_to_input(record: Dict[str, str]) -> Dict[str, str]:
        return {"task": record["task"]}

    def postprocess_answer(self, answer) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        return gsm_get_predict(answer)

    @staticmethod
    def record_to_target_answer(record: Dict[str, str]) -> str:
        return str(record["answer"])


def load_gsm8k_dataset(dataset_path: str) -> list[dict[str, str]]:
    dataset = JSONLReader.parse_file(dataset_path)
    return gsm_data_process(dataset)


def svamp_data_process(dataset):
    from dataset import svamp_dataset as _svamp_dataset

    list_data_dict = []
    for data in dataset:
        list_data_dict.append(
            {
                "task": _svamp_dataset._extract_question(data),
                "step": _svamp_dataset._extract_step(data),
                "answer": _svamp_dataset._normalize_numeric_answer(
                    _svamp_dataset._extract_answer(data)
                ),
            }
        )
    return list_data_dict


def multiarith_data_process(dataset):
    from dataset import multiarith_dataset as _multiarith_dataset

    list_data_dict = []
    for data in dataset:
        list_data_dict.append(
            {
                "task": _multiarith_dataset._extract_question(data),
                "step": _multiarith_dataset._extract_step(data),
                "answer": gsm_get_predict(_multiarith_dataset._extract_answer(data)),
            }
        )
    return list_data_dict


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
