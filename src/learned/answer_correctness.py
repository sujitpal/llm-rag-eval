import dspy
import json
import os

from typing import Dict

from .learning_utils import score_metric, optimize_prompt


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "answer_correctness.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
BEST_CONFIG = os.path.join(CONFIGS_DIR, "answer_correctness-best.json")
DEBUG = False


class AnswerToFacts(dspy.Signature):
    """ Extract 5-10 facts from the answer"""
    answer = dspy.InputField(desc="answer to extract facts from")
    facts = dspy.OutputField(desc="facts extracted from answer")


class FactGroupings(dspy.Signature):
    """ Classify each answer fact with respect to ground truth fact as follows:
        - TP (true positive): fact is present in both answer and ground truth
        - FP (false positive): fact is present in answer but not in ground truth
        - FN (false negative): fact is present in ground truth not in answer.
        Group the input facts as numbered lists under these 3 categories
    """
    facts_g = dspy.InputField(desc="facts from ground truth", format=str)
    facts_a = dspy.InputField(desc="facts from answer", format=str)
    fact_groups = dspy.OutputField(
        desc="groupings of facts as JSON string with keys TP, FP, FN and "
             "list of facts for each as values")


class AnswerCorrectness(dspy.Module):
    def __init__(self):
        super().__init__()
        self.fact_extractor = dspy.Predict(AnswerToFacts)
        self.fact_grouper = dspy.ChainOfThought(FactGroupings)

    def _parse_json_response(self, fact_groups: str) -> Dict[str, int]:
        fact_groups_j = json.loads(fact_groups
                                   .replace("```json", "")
                                   .replace("```", ""))
        return {k: len(v) for k, v in fact_groups_j.items()}

    def forward(self, answer, ground_truth):
        dspy.logger.debug(f"input answer: {answer}, ground_truth: {ground_truth}")
        facts_g = self.fact_extractor(answer=ground_truth).facts
        dspy.logger.debug(f"facts from ground truth: {facts_g}")
        facts_a = self.fact_extractor(answer=answer).facts
        dspy.logger.debug(f"facts from answer: {facts_a}")
        fact_groups = self.fact_grouper(
            facts_g=facts_g, facts_a=facts_a).fact_groups
        dspy.logger.debug(f"fact groups: {fact_groups}")
        group_counts = self._parse_json_response(fact_groups)
        tp = group_counts.get("TP", 0)
        fp = group_counts.get("FP", 0)
        fn = group_counts.get("FN", 0)
        score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0.0
        dspy.logger.debug(f"tp: {tp}, fp: {fp}, fn: {fn}, score: {score}")
        return dspy.Prediction(score=score)


def answer_correctness_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"answer correctness dataset {file_path} not found, "
            f"create it using generate_datasets.py first.")

    examples = []
    with open(file_path, "r") as fin:
        for line in fin:
            record = json.loads(line)
            answer = record["answer"]
            ground_truth = record["ideal_answer"]
            score = record["score"]
            examples.append(dspy.Example(
                answer=answer, ground_truth=ground_truth, score=str(score))
                .with_inputs("answer", "ground_truth"))
    return examples


def compute_answer_correctness(answer: str,
                               ideal_answer: str,
                               prompts_dict) -> float:
    try:
        answer_correctness_opt = prompts_dict["answer_correctness"]
    except KeyError:
        answer_correctness_opt = optimize_prompt("answer_correctness",
                                                 CONFIGS_DIR,
                                                 answer_correctness_dataset,
                                                 DATASET_FP,
                                                 score_metric,
                                                 AnswerCorrectness())
        prompts_dict["answer_correctness"] = answer_correctness_opt
    pred = answer_correctness_opt(answer=answer, ground_truth=ideal_answer)        
    return float(pred.score)
