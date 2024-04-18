import dspy
import json
import os

from typing import List

from .learning_utils import (
    list_to_string, string_to_bool, score_metric, optimize_prompt
)


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "context_precision.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")


class QuestionAnswerContextToUseful(dspy.Signature):
    """ Given a question, an answer to the question, and supporting context,
        provide a yes/no score indicating if the context was useful for
        answering the question."""
    question: str = dspy.InputField(desc="the question")
    answer: str = dspy.InputField(desc="answer to question")
    context: str = dspy.InputField(
        desc="supporting context used to answer question")
    score: str = dspy.OutputField(desc="yes or no")


class ContextPrecision(dspy.Module):
    def __init__(self):
        self.model = None
        self.usefulness_classifier = dspy.ChainOfThought(
            QuestionAnswerContextToUseful)
        
    def forward(self, question: str, answer: str,
                context: List[str]) -> str:
        dspy.logger.debug(f"input question: {question}, answer: {answer}, "
                          f"context: {context}")
        scores, weights = [], []
        for i, ctx in enumerate(context):
            pred = self.usefulness_classifier(question=question,
                                              answer=answer,
                                              context=ctx)
            scores.append(string_to_bool(pred.score, choices=["yes", "no"]))
        dspy.logger.debug(f"scores: {scores}")
        score = 0.0
        if len(scores) > 0:
            weights = [sum(scores[:i + 1]) / (i + 1) * scores[i]
                       for i in range(len(scores))]
            dspy.logger.debug(f"weights: {weights}")
            score = (sum(w * s for w, s in
                         zip(weights, scores)) / len(scores))
        dspy.logger.debug(f"score: {score}")
        return dspy.Prediction(score=str(score))


def context_precision_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"context precision dataset: {file_path} not found, "
            f"create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            question = record["question"]
            context = list_to_string(record["context"], style="number")
            answer = record["answer"]
            score = record["score"]
            examples.append(dspy.Example(
                question=question, context=context,
                answer=answer, score=str(score))
                .with_inputs("question", "context", "answer"))
    return examples


def compute_context_precision(question: str,
                              answer: str,
                              context: List[str],
                              prompts_dict):
    try:
        context_precision_opt = prompts_dict["context_precision"]
    except KeyError:
        context_precision_opt = optimize_prompt("context_precision",
                                                CONFIGS_DIR,
                                                context_precision_dataset,
                                                DATASET_FP,
                                                score_metric,
                                                ContextPrecision())
        prompts_dict["context_precision"] = context_precision_opt
    pred = context_precision_opt(question=question,
                                 answer=answer,
                                 context=context)
    return float(pred.score)
