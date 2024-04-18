import dspy
import json
import os

from typing import List

from .learning_utils import (
    list_to_string, string_to_list, string_to_bool,
    score_metric, optimize_prompt
)


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "faithfulness.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")


class QuestAnswerToFacts(dspy.Signature):
    """ Given a question-answer pair, generate a list of 3-5 facts
        from the answer
    """
    question: str = dspy.InputField(desc="a question")
    answer: str = dspy.InputField(desc="an answer")
    facts: str = dspy.OutputField(desc="a list of facts")


class ContextFactsToScore(dspy.Signature):
    """ Classify if fact can be inferred from context """
    context: str = dspy.InputField(desc="a context")
    fact: str = dspy.InputField(desc="a fact")
    score: bool = dspy.OutputField(
        desc="can fact be inferred from context? yes or no")


class Faithfulness(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(QuestAnswerToFacts)
        self.scorer = dspy.Predict(ContextFactsToScore)

    def forward(self, question: str, answer: str, context: str):
        dspy.logger.debug(f"input question: {question}, answer: {answer}, "
                          f"context: {context}")
        facts = self.extractor(question=question, answer=answer).facts
        dspy.logger.debug(f"facts: {facts}")
        scores = []
        for fact in string_to_list(facts):
            can_infer = self.scorer(context=context, fact=fact).score
            scores.append(string_to_bool(can_infer, ["yes", "no"]))
        dspy.logger.debug(f"scores: {scores}")
        score = sum(scores) / len(scores)
        dspy.logger.debug(f"score: {score}")
        return dspy.Prediction(score=str(score))


def faithfulness_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Faithfulness dataset: {file_path} not found, "
            "create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            question = record["question"]
            answer = record["answer"]
            context = list_to_string(record["context"], style="number")
            score = record["score"]
            examples.append(dspy.Example(
                question=question,
                answer=answer,
                context=context,
                score=str(score))
                .with_inputs("question", "answer", "context"))
    return examples


def compute_faithfulness(question: str,
                         answer: str,
                         context: List[str],
                         prompts_dict):
    try:
        faithfulness_opt = prompts_dict["faithfulness"]
    except KeyError:
        faithfulness_opt = optimize_prompt("faithfulness",
                                           CONFIGS_DIR,
                                           faithfulness_dataset,
                                           DATASET_FP,
                                           score_metric,
                                           Faithfulness())
        prompts_dict["faithfulness"] = faithfulness_opt
    pred = faithfulness_opt(
        question=question, answer=answer,
        context=list_to_string(context, style="number"))
    return float(pred.score)
