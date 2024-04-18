import dspy
import glob
import json
import numpy as np
import os
import shutil

from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.model_selection import train_test_split
from typing import List

from .learning_utils import string_to_list, list_to_string, string_to_bool


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "answer_relevance.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
BEST_CONFIG_FP = os.path.join(CONFIGS_DIR, "answer_relevance-best.json")


class AnswerContextToGenQuestions(dspy.Signature):
    """ Given the answer and context, generate 3 to 5 questions that can be 
        answered by the answer. """
    answer: str = dspy.InputField(desc="the answer")
    context: str = dspy.InputField(desc="the context of answer")
    gen_questions: str = dspy.OutputField(
        desc="list of questions that can be answered by answer")


class QuestionContextGenQToNonCommital(dspy.Signature):
    """ Given a question and its context, use only the context to generate
        an answer, and classify if the answer is commital or noncommital.
    """
    question: str = dspy.InputField(desc="the question")
    context: str = dspy.InputField(desc="the context of question")
    answer: str = dspy.OutputField(desc="the generated answer")
    noncommital: str = dspy.OutputField(desc="yes or no")


class AnswerRelevance(dspy.Module):
    def __init__(self, encoder):
        super().__init__()
        self.question_generator = dspy.Predict(
            AnswerContextToGenQuestions)
        self.answer_classifier = dspy.ChainOfThought(
            QuestionContextGenQToNonCommital)
        self.encoder = encoder

    def _cosine_similarity(self, source, targets):
        source = source.reshape(1, -1)
        sims = np.dot(source, targets.T) / (
            np.linalg.norm(source) * np.linalg.norm(targets, axis=1))
        return np.mean(sims)

    def _compute_score(self, q_list: List[str]):
        embeddings = self.encoder.embed_documents(q_list)
        E = np.array(embeddings)
        source, targets = E[0, :], E[1:, :]
        if len(targets) == 0:
            return 0.0
        return self._cosine_similarity(source, targets)
    
    def forward(self, question: str, answer: str, context: str):
        dspy.logger.debug(f"input question: {question}, answer: {answer}, "
                          f"context: {context}")
        gen_questions = self.question_generator(
            answer=answer, context=context).gen_questions
        dspy.logger.debug(f"gen_questions: {gen_questions}")
        q_list = [question]
        for gen_q in string_to_list(gen_questions):
            ans_cls = self.answer_classifier(question=gen_q, context=context)
            noncommital = ans_cls.noncommital
            if not string_to_bool(noncommital, choices=["yes", "no"]):
                q_list.append(gen_q)
        dspy.logger.debug(f"q_list: {q_list}")
        score = self._compute_score(q_list)
        dspy.logger.debug(f"score: {score}")
        return dspy.Prediction(score=str(score))


def answer_relevance_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"answer relevance dataset: {file_path} not found, "
            f"create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            question = record["question"]
            answer = record["answer"]
            context = list_to_string(record["context"], style="number")
            score = record["score"]
            examples.append(dspy.Example(
                question=question, answer=answer,
                context=context, score=score)
                .with_inputs("question", "answer", "context"))
    return examples


def answer_relevance_metric(example, pred, trace=None):
    if trace is None:
        return 1.0 - abs(float(example.score) - float(pred.score))
    else:
        return float(pred.score)


def optimize_prompt(encoder: GoogleGenerativeAIEmbeddings):
    config_paths = glob.glob(
        os.path.join(CONFIGS_DIR, "answer_relevance-*.json"))

    if len(config_paths) == 0:
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=answer_relevance_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1)
        examples = answer_relevance_dataset(DATASET_FP)
        trainset, devset = train_test_split(
            examples, test_size=0.3, random_state=42)
        print(f"fact extractor dataset sizes: "
              f"{len(trainset)}, {len(devset)}, total: {len(examples)}")

        print("--- training ---")
        answer_relevance = AnswerRelevance(encoder=encoder)
        answer_relevance_opt = teleprompter.compile(
            answer_relevance, trainset=trainset)
        ensemble = [prog for *_, prog in
                    answer_relevance_opt.candidate_programs[:4]]
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(CONFIGS_DIR, f"answer_relevance-{idx}.json")
            config_paths.append(config_path)
            prog.save(config_path)

        print("--- evaluation ---")
        evaluate = Evaluate(devset=devset, metric=answer_relevance_metric,
                            num_threads=1, display_progress=True)
        scores = [evaluate(prog) for prog in ensemble]
        print(f"Evaluation scores: {scores}")
        best_prompt_id = np.argmax(scores)
        shutil.copy(config_paths[best_prompt_id], BEST_CONFIG_FP)

    prog = AnswerRelevance(encoder)
    prog.load(BEST_CONFIG_FP)
    return prog


def compute_answer_relevance(question: str,
                             context: List[str],
                             answer: str,
                             prompts_dict, 
                             encoder):
    try:
        answer_relevance_opt = prompts_dict["answer_relevance"]
    except KeyError:
        answer_relevance_opt = optimize_prompt(encoder)
        prompts_dict["answer_relevance"] = answer_relevance_opt
    context_str = list_to_string(context, style="number")
    pred = answer_relevance_opt(
        question=question, answer=answer, context=context_str)
    return float(pred.score)
