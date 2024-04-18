import dspy
import glob
import re
import numpy as np
import os
import shutil

from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import train_test_split
from typing import Callable, List


STYLE_TO_CHAR = {
    "dash": "-",
    "star": "*",
}
NUM_BULLET_PATTERM = re.compile(r"^\d+(\.)?\s(.*?)$")
GEN_BULLET_PATTERN = re.compile(r"^[*-]\s(.*?)$")


def string_to_list(xs: str, style: str = "dash") -> List[str]:
    """ Convert a string with embedded newlines to a list of strings
        separated by newlines.

        :param xs: string with embedded newlines
        :param style: bullet style to strip (dash, star, number)
        :return: a list of strings
    """
    stripped_xs = []
    for x in xs.split("\n"):
        if NUM_BULLET_PATTERM.match(x):
            stripped_x = NUM_BULLET_PATTERM.match(x).group(2)
        elif GEN_BULLET_PATTERN.match(x):
            stripped_x = GEN_BULLET_PATTERN.match(x).group(1)
        else:
            stripped_x = x
        stripped_xs.append(stripped_x)
    return stripped_xs


def list_to_string(xs: List[str], style: str = "dash") -> str:
    """ Convert a list of strings to a string with embedded newlines.
        Strips (known) bullet chars off each string element and
        adds in the appropriate bullet specified by the style.

        :param xs: list of strings
        :param style: bullet style to strip (dash, star, number)
        :return: a string with embedded newlines
    """
    stripped_xs = []
    for i, x in enumerate(xs):
        if NUM_BULLET_PATTERM.match(x):
            stripped_x = NUM_BULLET_PATTERM.match(x).group(2)
        elif GEN_BULLET_PATTERN.match(x):
            stripped_x = GEN_BULLET_PATTERN.match(x).group(1)
        else:
            stripped_x = x
        if style == "number":
            stripped_x = f"{i+1}. {stripped_x}"
        elif style in ["dash", "star"]:
            bullet = STYLE_TO_CHAR.get(style, "-")
            stripped_x = f"{bullet} {stripped_x}"
        else:
            pass
        stripped_xs.append(stripped_x)
    return "\n".join(stripped_xs)


def string_to_bool(s: str, choices: List[str]) -> bool:
    """ Convert a string to a boolean given list of choices representing
        True and False response

        :param s: string to convert
        :param choices: list of strings ["True", "False]
        :return: boolean
    """
    matches = [i for i, c in enumerate(choices)
               if re.search(c.lower(), s.lower()) is not None]
    if len(matches) == 0:
        return False
    else:
        return True if matches[0] == 0 else False


def string_to_bool_array(s: str, choices: List[str]) -> List[bool]:
    """ Convert a string to a list of booleans given list of choices
        representing True and False response

        :param s: string to convert
        :param choices: list of strings ["True", "False]
        :return: counts for each choice found
    """
    patterns = [re.compile(c) for c in choices]
    counts = [len(re.findall(p, s)) for p in patterns]
    return counts


def strip_newlines(s: str) -> str:
    """ Strip newlines from a string

        :param s: string to strip
        :return: string with newlines stripped
    """
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def score_metric(example, pred, trace=None):
    """ Common score metric used in all learning metrics """
    if trace is None:
        return 1.0 - abs(float(example.score) - float(pred.score))
    else:
        return float(pred.score)    # inference


def optimize_prompt(metric_name: str,
                    config_dir: str,
                    dataset_generator_fn: Callable,
                    dataset_fp: str,
                    score_metric: Callable,
                    metric_instance: dspy.Module):

    config_fn_pattern = f"{metric_name}-*.json"
    config_paths = glob.glob(os.path.join(config_dir, config_fn_pattern))
    best_config_fp = os.path.join(config_dir, f"{metric_name}-best.json")

    if len(config_paths) == 0:
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=score_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1
        )
        examples = dataset_generator_fn(dataset_fp)
        trainset, devset = train_test_split(examples, test_size=0.3,
                                            random_state=42)
        print(
            f"fact extractor dataset sizes: "
            f"{len(trainset)}, {len(devset)}, total: {len(examples)}")

        print("--- training ---")
        faithfulness = metric_instance
        faithfulness_opt = teleprompter.compile(
            faithfulness, trainset=trainset)
        ensemble = [prog for *_, prog in
                    faithfulness_opt.candidate_programs[:4]]
        
        os.makedirs(config_dir, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(
                config_dir, f"{metric_name}-{idx}.json")
            config_paths.append(config_path)
            prog.save(config_path)

        print("--- evaluation ---")
        evaluate = Evaluate(devset=devset, metric=score_metric,
                            num_threads=1, display_progress=True)
        scores = [evaluate(prog) for prog in ensemble]
        print(f"Evaluation scores: {scores}")
        best_prompt_id = np.argmax(scores)
        shutil.copy(config_paths[best_prompt_id], best_config_fp)

    prog = metric_instance
    prog.load(best_config_fp)
    return prog


def clean_up_log_files():
    log_files = ["openai_usage.log",
                 "azure_openai_usage.log",
                 "assertion.log"]
    for log_file in log_files:
        if os.path.exists(log_file):
            os.remove(log_file)
