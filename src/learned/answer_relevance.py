import dspy
import json
import numpy as np
import os

from typing import List

from .learning_utils import (
    string_to_list, list_to_string, string_to_bool,
    score_metric, optimize_prompt
)


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "answer_relevance.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")


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


def compute_answer_relevance(question: str,
                             context: List[str],
                             answer: str,
                             prompts_dict, 
                             encoder):
    try:
        answer_relevance_opt = prompts_dict["answer_relevance"]
    except KeyError:
        answer_relevance_opt = optimize_prompt("answer_relevance",
                                               CONFIGS_DIR,
                                               answer_relevance_dataset,
                                               DATASET_FP,
                                               score_metric,
                                               AnswerRelevance(encoder=encoder))
        prompts_dict["answer_relevance"] = answer_relevance_opt
    dspy.logger.debug(f"context: {context}")
    context_str = list_to_string(context, style="number")
    pred = answer_relevance_opt(
        question=question, answer=answer, context=context_str)
    return float(pred.score)
