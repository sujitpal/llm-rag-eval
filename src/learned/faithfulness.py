import argparse
import dspy
import glob
import json
import numpy as np
import os
import shutil

from dotenv import find_dotenv, load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import train_test_split

from .learning_utils import list_to_string, string_to_list, string_to_bool


DATA_DIR = "../data"
RESOURCE_DIR = "../resources"

DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
DATASET_FP = os.path.join(DATASET_DIR, "faithfulness.jsonl")
CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
BEST_CONFIG = os.path.join(CONFIGS_DIR, "faithfulness-best.json")


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
        facts = self.extractor(question=question, answer=answer).facts
        scores = []
        for fact in string_to_list(facts):
            can_infer = self.scorer(context=context, fact=fact).score
            scores.append(string_to_bool(can_infer, ["yes", "no"]))
        score = sum(scores) / len(scores)
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


def faithfulness_metric(example, pred, trace=None):
    if trace is None:
        return 1.0 - abs(float(example.score) - float(pred.score))
    else:
        return float(pred.score)     # for inference


def optimize_prompt():

    config_paths = glob.glob(os.path.join(CONFIGS_DIR, "faithfulness-*.json"))

    if len(config_paths) == 0:
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=faithfulness_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1
        )
        examples = faithfulness_dataset(DATASET_FP)
        trainset, devset = train_test_split(examples, test_size=0.3,
                                            random_state=42)
        print(f"fact extractor dataset sizes: "
              f"{len(trainset)}, {len(devset)}, total: {len(examples)}")

        print("--- training ---")
        faithfulness = Faithfulness()
        faithfulness_opt = teleprompter.compile(
            faithfulness, trainset=trainset)
        ensemble = [prog for *_, prog in
                    faithfulness_opt.candidate_programs[:4]]
        
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        for idx, prog in enumerate(ensemble):
            config_path = os.path.join(
                CONFIGS_DIR, f"faithfulness-{idx}.json")
            prog.save(config_path)

        print("--- evaluation ---")
        evaluate = Evaluate(devset=devset, metric=faithfulness_metric,
                            num_threads=1, display_progress=True)
        scores = [evaluate(prog) for prog in ensemble]
        print(f"Evaluation scores: {scores}")
        best_prompt_id = np.argmax(scores)
        shutil.copy(config_paths[best_prompt_id], BEST_CONFIG)

    prog = Faithfulness()
    prog.load(BEST_CONFIG)
    return prog


def compute_faithfulness(question: str,
                         answer: str,
                         context: str,
                         prompts_dict,
                         model):
    try:
        faithfulness_opt = prompts_dict["faithfulness"]
    except KeyError:
        faithfulness_opt = optimize_prompt()
        prompts_dict["faithfulness"] = faithfulness_opt
    pred = faithfulness_opt(
        question=question, answer=answer, context=context)
    return float(pred.score)


# if True:

#     _ = load_dotenv(find_dotenv())

#     gemini = dspy.Google("models/gemini-1.0-pro",
#                          api_key=os.environ["GOOGLE_API_KEY"],
#                          max_output_tokens=1024,
#                          temperature=0.3)
#     dspy.settings.configure(lm=gemini)

#     faithfulness_opt = optimize_prompt()
    
#     question = "What are the global implications of the USA Supreme Court ruling on abortion?"
#     answer = """
# The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. Here are some potential implications:

# 1. Influence on other countries: The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws. It can provide legal arguments and reasoning that advocates for reproductive rights can use to challenge restrictive abortion laws in their respective jurisdictions.

# 2. Strengthening of global reproductive rights movements: A favorable ruling by the Supreme Court can energize and empower reproductive rights movements worldwide. It can serve as a rallying point for activists and organizations advocating for women's rights, leading to increased mobilization and advocacy efforts globally.

# 3. Counteracting anti-abortion movements: Conversely, a ruling that restricts abortion rights can embolden anti-abortion movements globally. It can provide legitimacy to their arguments and encourage similar restrictive measures in other countries, potentially leading to a rollback of existing reproductive rights.

# 4. Impact on international aid and policies: The Supreme Court's ruling can influence international aid and policies related to reproductive health. It can shape the priorities and funding decisions of donor countries and organizations, potentially leading to increased support for reproductive rights initiatives or conversely, restrictions on funding for abortion-related services.

# 5. Shaping international human rights standards: The ruling can contribute to the development of international human rights standards regarding reproductive rights. It can influence the interpretation and application of existing human rights treaties and conventions, potentially strengthening the recognition of reproductive rights as fundamental human rights globally.

# 6. Global health implications: The Supreme Court's ruling can have implications for global health outcomes, particularly in countries with restrictive abortion laws. It can impact the availability and accessibility of safe and legal abortion services, potentially leading to an increase in unsafe abortions and related health complications.

# It is important to note that the specific implications will depend on the nature of the Supreme Court ruling and the subsequent actions taken by governments, activists, and organizations both within and outside the United States.
# """
#     context = """
# "- In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.
# - This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.
# - The states with the most restrictive abortion laws have the weakest maternal health support, higher maternal death rates, and higher child poverty rates.
# - The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally and the aid it funds.
# - SRR organizations and activists across the world have expressed fear about the ruling laying the groundwork for anti-abortion legislative and policy attacks in other countries.
# - Advocates have also observed the ruling's impact on progressive law reform and the stalling of the adoption and enforcement of abortion guidelines in certain African countries.
# - The ruling has created a chilling effect in international policy spaces, emboldening anti-abortion state and non-state actors to undermine human rights protections."""

#     score = compute_faithfulness(question, answer, context,
#                                  faithfulness_opt, gemini)
#     print(score)

#     # config_paths = glob.glob(os.path.join(CONFIGS_DIR, "faithfulness-*.json"))
#     # if len(config_paths) == 0:
#     #     teleprompter = BootstrapFewShotWithRandomSearch(
#     #         metric=faithfulness_metric,
#     #         max_bootstrapped_demos=2,
#     #         max_labeled_demos=2,
#     #         num_threads=1
#     #     )
#     #     examples = faithfulness_dataset(DATASET_FP)
#     #     trainset, devset = train_test_split(examples, test_size=0.3,
#     #                                         random_state=42)
#     #     print(f"fact extractor dataset sizes: "
#     #             f"{len(trainset)}, {len(devset)}, total: {len(examples)}")

#     #     print("--- training ---")
#     #     faithfulness = Faithfulness()
#     #     faithfulness_opt = teleprompter.compile(
#     #         faithfulness, trainset=trainset)
#     #     ensemble = [prog for *_, prog in
#     #                 faithfulness_opt.candidate_programs[:4]]
        
#     #     os.makedirs(CONFIGS_DIR, exist_ok=True)
#     #     for idx, prog in enumerate(ensemble):
#     #         config_path = os.path.join(
#     #             CONFIGS_DIR, f"faithfulness-{idx}.json")
#     #         prog.save(config_path)

#     #     print("--- evaluation ---")
#     #     evaluate = Evaluate(devset=devset, metric=faithfulness_metric,
#     #                         num_threads=1, display_progress=True)
#     #     scores = [evaluate(prog) for prog in ensemble]
#     #     print(f"Evaluation scores: {scores}")

