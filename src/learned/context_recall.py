import dspy
import json
import nltk
import numpy as np
import os

from typing import List

from .learning_utils import string_to_bool_array, score_metric, optimize_prompt


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "context_recall.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
BEST_CONFIG = os.path.join(CONFIGS_DIR, "context_recall-best.json")


class ContextItemAnswerToScore(dspy.Signature):
    """ Given a context item and an answer, for each sentence in the answer,
        classify if the sentence can be attributed to the context item. """
    answer = dspy.InputField(desc="the answer", format=str)
    context_item = dspy.InputField(desc="the context item")
    scores = dspy.OutputField(
        desc="yes/no for each answer sentence if it is attributale to context")


class ContextRecall(dspy.Module):
    def __init__(self):
        super().__init__()
        self.attrib_clf = dspy.ChainOfThought(ContextItemAnswerToScore)

    def forward(self, context: List[str], answer: str):
        dspy.logger.debug(f"input context: {context}, answer: {answer}")
        answer_sents = [sent for sent
                        in nltk.sent_tokenize(answer.replace("\n", ""))
                        if len(sent.strip()) > 0][0:10]
        dspy.logger.debug(f"answer sentences: {answer_sents}")
        scores = []
        for context_item in context:
            if len(context_item.strip()) < 10:
                continue
            ctx_score = 0.0
            try:
                ctx_scores = self.attrib_clf(
                    answer=answer_sents,
                    context_item=context_item).scores
                num_pos, num_neg = string_to_bool_array(
                    ctx_scores, choices=["yes", "no"])
                if num_pos + num_neg > 0:
                    ctx_score = num_pos / (num_pos + num_neg)
            except Exception:
                pass
            # print(f"context: {context_item}, score: {ctx_score}")
            scores.append(ctx_score)
        dspy.logger.debug(f"scores: {scores}")
        score = 0.0
        if len(scores) > 0:
            score = np.mean(scores)
        dspy.logger.debug(f"score: {score}")
        return dspy.Prediction(score=str(score))


def context_recall_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"context recall dataset: {file_path} not found, "
            "create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            answer = record["answer"]
            context = record["context"]
            score = record["score"]
            examples.append(dspy.Example(
                answer=answer,
                context=context,
                score=str(score))
                .with_inputs("answer", "context"))
    return examples


def compute_context_recall(context: List[str],
                           answer: str,
                           prompts_dict):
    try:
        context_recall_opt = prompts_dict["context_recall"]
    except KeyError:
        context_recall_opt = optimize_prompt("context_recall",
                                             CONFIGS_DIR,
                                             context_recall_dataset,
                                             DATASET_FP,
                                             score_metric,
                                             ContextRecall())
        prompts_dict["context_recall"] = context_recall_opt
    pred = context_recall_opt(context=context, answer=answer)
    return float(pred.score)
