import dspy
import glob
import json
import nltk
import numpy as np
import os
import shutil
import time

from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import train_test_split
from typing import List

from .learning_utils import list_to_string, string_to_bool, score_metric


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "context_relevance.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
BEST_CONFIG = os.path.join(CONFIGS_DIR, "context_relevance-best.json")


class QuestionCtxSentToScore(dspy.Signature):
    """ Given a question and a sentence from the context, classify
        if sentence is absolutely necessary to answer question
    """
    question: str = dspy.InputField(desc="the question")
    ctx_sent: str = dspy.InputField(desc="a sentence from the context")
    score: float = dspy.OutputField(desc="yes or no")


class ContextRelevance(dspy.Module):
    def __init__(self):
        super().__init__()
        self.nec_classifier = dspy.ChainOfThought(QuestionCtxSentToScore)

    def forward(self, question: str, context: List[str]):
        dspy.logger.debug(f"input question: {question}, context: {context}")
        ctx_scores = []
        for ctx in context:
            sent_scores = []
            for ctx_sent in nltk.sent_tokenize(ctx):
                score = self.nec_classifier(question=question,
                                            ctx_sent=ctx_sent).score
                sent_scores.append(string_to_bool(score, choices=["yes", "no"]))
            if len(sent_scores) == 0:
                ctx_scores.append(0.0)
            else:
                ctx_scores.append(sum(sent_scores) / len(sent_scores))
            # to prevent ResourceExhaustedError
            time.sleep(0.3)
        dspy.logger.debug(f"context scores: {ctx_scores}")
        score = 0.0
        if len(ctx_scores) > 0:
            score = sum(ctx_scores) / len(ctx_scores)
        dspy.logger.debug(f"score: {score}")
        return dspy.Prediction(score=str(score))


def context_relevance_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"context relevance dataset: {file_path} not found, "
            "create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            question = record["question"]
            context = record["context"]
            score = record["score"]
            examples.append(dspy.Example(
                question=question,
                context=list_to_string(context),
                score=str(score)
            ).with_inputs("question", "context"))
    return examples


def optimize_prompt():
    config_paths = glob.glob(os.path.join(CONFIGS_DIR, "context_relevance-*.json"))
    if len(config_paths) == 0:
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=score_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1)
        examples = context_relevance_dataset(DATASET_FP)
        trainset, devset = train_test_split(
            examples, test_size=0.3, random_state=42)
        print(f"fact extractor dataset sizes: "
              f"{len(trainset)}, {len(devset)}, total: {len(examples)}")

        print("--- training ---")
        context_relevance = ContextRelevance()
        context_relevance_opt = teleprompter.compile(
            context_relevance, trainset=trainset)
        ensemble = [prog for *_, prog in
                    context_relevance_opt.candidate_programs[:4]]

        os.makedirs(CONFIGS_DIR, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(
                CONFIGS_DIR, f"context_relevance-{idx}.json")
            config_paths.append(config_path)
            prog.save(config_path)

        print("--- evaluation ---")
        evaluate = Evaluate(devset=devset, metric=score_metric,
                            num_threads=1, display_progress=True)
        scores = [evaluate(prog) for prog in ensemble]
        print(f"Evaluation scores: {scores}")
        best_prompt_id = np.argmax(scores)
        shutil.copy(config_paths[best_prompt_id], BEST_CONFIG)

    prog = ContextRelevance()
    prog.load(BEST_CONFIG)
    return prog


def compute_context_relevance(question: str,
                              context: List[str],
                              prompts_dict):
    try:
        context_relevance_opt = prompts_dict["context_relevance"]
    except KeyError:
        context_relevance_opt = optimize_prompt()
        prompts_dict["context_relevance"] = context_relevance_opt
    pred = context_relevance_opt(question=question, context=context)
    return float(pred.score)
