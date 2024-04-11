import dspy
import glob
import json
import logging
import numpy as np
import os
import re
import torch

from dotenv import find_dotenv, load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from nltk.translate import bleu
from sklearn.model_selection import train_test_split
from typing import List

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

DATA_DIR = "../../data"
DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
RESOURCE_DIR = "../../resources"
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
CHECKPOINTS_DIR = os.path.join(RESOURCE_DIR, "checkpoints", "faithfulness_2")

GOLDSET_FP = os.path.join(DATA_DIR, "goldset_ragas.jsonl")
REPORTS_DIR = os.path.join(DATA_DIR, "dspy-reports")
REPORT_FP = os.path.join(REPORTS_DIR, "faithfulness.tsv")


def fact_extractor_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Faithfulness dataset: {file_path} not found, "
            "create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            answer = record["answer"]
            statements = "\n- ".join(record["statements"])
            score = bleu([answer], statements)
            examples.append(dspy.Example(
                answer=answer,
                score=float(score))
                .with_inputs("answer"))
    return examples


class AnswerToFact(dspy.Signature):
    """ Extracts facts from an answer """
    answer: str = dspy.InputField()
    facts: str = dspy.OutputField(desc="a list of facts")


class FactExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(AnswerToFact)

    def forward(self, answer: str):
        return self.extractor(answer=answer)


def bleu_metric(example, pred, trace=None):
    if trace is None:
        return bleu([example.answer], pred.facts)
    else:
        return example.score


def optimize_or_get_fact_extractor(evaluate_model: bool = False):

    config_paths = glob.glob(os.path.join(CONFIGS_DIR, "faithfulness_1-*.json"))

    if len(config_paths) == 0:
        # not optimized, optimize prompt and save
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=bleu_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_threads=1
        )

        examples = fact_extractor_dataset(
            os.path.join(DATASET_DIR, "faithfulness.jsonl"))
        trainset, devset = train_test_split(examples, test_size=0.3,
                                            random_state=42)
        logging.info(f"fact extractor dataset sizes: "
                    f"{len(trainset)}, {len(devset)}")

        # train the prompt
        fact_extractor = FactExtractor()
        fact_extractor_opt = teleprompter.compile(
            fact_extractor, trainset=trainset)
        ensemble = [prog for *_, prog in
                    fact_extractor_opt.candidate_programs[:4]]

        os.makedirs(CONFIGS_DIR, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(CONFIGS_DIR,
                                       f"faithfulness_1-{idx}.json")
            prog.save(config_path)

        # evaluate the prompt
        if evaluate_model:
            evaluate = Evaluate(devset=devset, metric=bleu_metric,
                                num_threads=1, display_progress=True,
                                display_table=1)
            scores = [evaluate(prog) for prog in ensemble]
            logging.info(f"Evaluation scores: {scores}")

    fact_extractor_ensemble = []
    for config_path in config_paths:
        prog = FactExtractor()
        prog.load(config_path)
        fact_extractor_ensemble.append(prog)

    return fact_extractor_ensemble


def get_statements_from_answer(
        answer: str, fact_extractor_ensemble: List[dspy.Module]
        ) -> List[str]:
    scores, facts_list = [], []
    for idx, prog in enumerate(fact_extractor_ensemble):
        pred = prog(answer=answer)
        facts = pred.facts
        score = bleu([answer], facts)
        scores.append(score)
        facts_list.append(facts)
        logging.debug(f"score[{idx}]: {score:.2f} - {pred.facts[:100]}...")
    best_facts = facts_list[np.argmax(scores)]
    facts_list = best_facts.split("\n")
    facts_list = [re.sub(r"^-\s", "", f) for f in facts_list]
    return facts_list


def nli_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Faithfulness dataset: {file_path} not found, "
            "create it with generate_datasets.py first.")
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            contexts = record["context"]
            facts = record["statements"]
            entailments = record["entailments"]
            for context, fact_entailment in zip(contexts, entailments):
                for fact, entailment in zip(facts, fact_entailment):
                    examples.append(dspy.Example(
                        context=context,
                        fact=fact,
                        score=entailment)
                        .with_inputs("context", "fact"))
    return examples


class ContextToFact(dspy.Signature):
    """ Return 0 or 1 based on whether the fact can be inferred from
        the context.
    """
    context: str = dspy.InputField(desc="Context")
    fact: str = dspy.InputField(desc="A fact to check")
    score: int = dspy.OutputField(
        desc="Indicate if the fact can be inferred from the context",
        choices=[0, 1])


class NLIChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.checker = dspy.ChainOfThought(ContextToFact)

    def forward(self, context: str, fact: str):
        return self.checker(context=context, fact=fact)


def nli_exact_match(example, pred, trace=None):
    if trace is None:
        pred_score = 0 if pred.score == "No" else 1
        return pred_score == example.score
    else:
        return example.score


def fine_tune__or_get_nli_checker(evaluate_model: bool = True):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, T5 fine-tuning needs GPU.")

    if not os.path.exists(CHECKPOINTS_DIR):
        # fine-tune model from data
        # :NOTE: maybe should have gone through an optimized prompt
        # and used that as a teacher?

        nli_dataset_path = os.path.join(DATASET_DIR, "faithfulness.jsonl")
        # :NOTE: just using 50 examples because Gemini complains about
        # ResourceExhaustion and asks to increase quota
        examples = nli_dataset(nli_dataset_path)[0:50]
        trainset, devset = train_test_split(examples, test_size=0.3,
                                            random_state=42)
        logging.info(f"NLI checker dataset size: {len(trainset)}, {len(devset)}")

        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        config = dict(target="t5-base",
                      path_prefix=CHECKPOINTS_DIR,
                      epochs=2,
                      bf16=False,
                      bsize=6,
                      accumsteps=2,
                      lr=5e-5)
        teleprompter = BootstrapFinetune(metric=nli_exact_match)
        nli_checker_opt = teleprompter.compile(NLIChecker(),
                                               trainset=trainset,
                                               **config)
        
        if evaluate_model:
            evaluate = Evaluate(devset=devset, metric=nli_exact_match,
                                num_threads=1, display_progress=True,
                                display_table=1)
            scores = evaluate(nli_checker_opt)
            logging.info(f"Evaluation scores: {scores}")

    # :NOTE: you will need to manually save the best checkpoint
    # (from the training logs) to the CHECKPOINTS_DIR folder
    t5_model = dspy.HFModel(checkpoint=CHECKPOINTS_DIR, model="t5-base")
    nli_checker = NLIChecker()
    for p in nli_checker.predictors():
        p.lm = t5_model
        p.activated = False

    return nli_checker


def get_entailments_from_context(nli_checker: dspy.Module,
                                 contexts: List[str],
                                 facts: List[str]
                                 ) -> List[List[int]]:
    entailments = []
    for context in contexts:
        fact_entailments = []
        for fact in facts:
            pred = nli_checker(context=context, fact=fact)
            fact_entailments.append(pred.score)
        entailments.append(fact_entailments)
    return entailments


def compute_fact_score(fact_entailments: List[str]) -> float:
    print("fact_entailments:", fact_entailments)
    fact_scores = []
    for entailment in fact_entailments:
        fact_score = 1 if entailment in ["Yes", "True", "1"] else 0
        fact_scores.append(fact_score)
    return np.mean(fact_scores)


if __name__ == "__main__":

    _ = load_dotenv(find_dotenv())

    gemini = dspy.Google("models/gemini-1.0-pro",
                         api_key=os.environ["GOOGLE_API_KEY"],
                         max_output_tokens=1024,
                         temperature=0.3)
    dspy.settings.configure(lm=gemini)

    fact_extractor = optimize_or_get_fact_extractor(evaluate_model=True)
    nli_checker = fine_tune__or_get_nli_checker(evaluate_model=True)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(GOLDSET_FP, "r", encoding="utf-8") as fin, \
         open(REPORT_FP, "w", encoding="utf-8") as fout:
    
        fout.write("\t".join(["#QID", "FAITHFULNESS"]) + "\n")
        for line in fin:
            record = json.loads(line)
            id = record["id"]
            question = record["query"]
            logging.info(f"Processing question ({id}): {question}...")
            contexts = [ctx["chunk_text"] for ctx in record["context"]]
            answer = record["predicted_answer"]
            ideal_answer = record["ideal_answer"]

            facts = get_statements_from_answer(answer, fact_extractor)
            entailments = get_entailments_from_context(
                nli_checker, contexts, facts)

            scores = []
            for fact_entailments in entailments:
                score = compute_fact_score(fact_entailments)
                scores.append(score)
            metric_value = np.mean(scores)
            fout.write(f"{id}\t{metric_value:.3f}\n")
