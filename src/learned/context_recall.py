import dspy
import glob
import json
import nltk
import numpy as np
import os
import shutil

from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import train_test_split
from typing import List

from .learning_utils import string_to_bool_array


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
        answer_sents = [sent for sent
                        in nltk.sent_tokenize(answer.replace("\n", ""))
                        if len(sent.strip()) > 0][0:10]
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
        score = 0.0
        if len(scores) > 0:
            score = np.mean(scores)
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


def context_recall_metric(example, pred, trace=None):
    if trace is None:
        return 1.0 - abs(float(example.score) - float(pred.score))
    else:   
        return float(pred.score)     # for inference


def optimize_prompt():

    config_paths = glob.glob(os.path.join(CONFIGS_DIR, "context_recall-*.json"))

    if len(config_paths) == 0:
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=context_recall_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1
        )
        examples = context_recall_dataset(DATASET_FP)
        trainset, devset = train_test_split(examples, test_size=0.3,
                                            random_state=42)
        print(f"fact extractor dataset sizes: "
              f"{len(trainset)}, {len(devset)}, total: {len(examples)}")

        print("--- training ---")
        context_recall = ContextRecall()
        context_recall_opt = teleprompter.compile(
            context_recall, trainset=trainset)
        ensemble = [prog for *_, prog in
                    context_recall_opt.candidate_programs[:4]]
        
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(
                CONFIGS_DIR, f"context_recall-{idx}.json")
            config_paths.append(config_path)
            prog.save(config_path)

        print("--- evaluation ---")
        evaluate = Evaluate(devset=devset, metric=context_recall_metric,
                            num_threads=1, display_progress=True)
        scores = [evaluate(prog) for prog in ensemble]
        print(f"Evaluation scores: {scores}")
        best_prompt_id = np.argmax(scores)
        shutil.copy(config_paths[best_prompt_id], BEST_CONFIG)

    prog = ContextRecall()
    prog.load(BEST_CONFIG)
    return prog


def compute_context_recall(context: List[str],
                           answer: str,
                           prompts_dict):
    try:
        context_recall_opt = prompts_dict["context_recall"]
    except KeyError:
        context_recall_opt = optimize_prompt()
        prompts_dict["context_recall"] = context_recall_opt
    pred = context_recall_opt(context=context, answer=answer)
    return float(pred.score)
