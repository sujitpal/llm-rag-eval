import dspy
import glob
import json
import numpy as np
import os
import shutil

from dotenv import load_dotenv, find_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.model_selection import train_test_split
from typing import List

from .learning_utils import string_to_list, list_to_string, string_to_bool


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "context_precision.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
BEST_CONFIG_FP = os.path.join(CONFIGS_DIR, "context_precision-best.json")


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
        
    def forward(self, question: str, answer: str, context: str) -> str:
        ctx_list = string_to_list(context)
        scores, weights = [], []
        for i, ctx in enumerate(ctx_list):
            pred = self.usefulness_classifier(question=question,
                                              answer=answer,
                                              context=ctx)
            scores.append(string_to_bool(pred.score, choices=["yes", "no"]))
        score = 0.0
        if len(scores) > 0:
            weights = [sum(scores[:i + 1]) / (i + 1) * scores[i]
                       for i in range(len(scores))]
            score = (sum(w * s for w, s in
                         zip(weights, scores)) / len(scores))
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


def context_precision_metric(example, pred, trace=None):
    if trace is None:
        return 1.0 - abs(float(example.score) - float(pred.score))
    else:
        return float(pred.score)  # inference


def optimize_prompt():
    config_paths = glob.glob(os.path.join(CONFIGS_DIR, "context_precision-*.json"))
    if len(config_paths) == 0:

        teleprompt = BootstrapFewShotWithRandomSearch(
            metric=context_precision_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1)

        examples = context_precision_dataset(DATASET_FP)
        trainset, devset = train_test_split(
            examples, test_size=0.3, random_state=42)
        print(f"context precision dataset sizes: "
              f"{len(trainset)}, {len(devset)}, total: {len(examples)}")
        
        print("--- training ---")
        context_precision = ContextPrecision()
        context_precision_opt = teleprompt.compile(
            context_precision, trainset=trainset)
        ensemble = [prog for *_, prog in
                    context_precision_opt.candidate_programs[:4]]
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(
                CONFIGS_DIR, f"context_precision-{idx}.json")
            config_paths.append(config_path)
            prog.save(config_path)

        print("--- evaluation ---")
        evaluate = Evaluate(devset=devset, metric=context_precision_metric,
                            num_threads=1, display_progress=True)
        scores = [evaluate(prog) for prog in ensemble]
        print(f"Evaluation scores: {scores}")
        best_prompt_id = np.argmax(scores)
        shutil.copy(config_paths[best_prompt_id], BEST_CONFIG_FP)

    prog = ContextPrecision()
    prog.load(BEST_CONFIG_FP)
    return prog


def compute_context_precision(question: str,
                              answer: str,
                              context: str,
                              prompts_dict):
    try:
        context_precision_opt = prompts_dict["context_precision"]
    except KeyError:
        context_precision_opt = optimize_prompt()
        prompts_dict["context_precision"] = context_precision_opt
    pred = context_precision_opt(question=question,
                                 answer=answer,
                                 context=context)
    return float(pred.score)
