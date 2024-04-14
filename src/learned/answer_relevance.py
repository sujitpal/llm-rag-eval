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
        gen_questions = self.question_generator(
            answer=answer, context=context).gen_questions
        q_list = [question]
        for gen_q in string_to_list(gen_questions):
            ans_cls = self.answer_classifier(question=gen_q, context=context)
            noncommital = ans_cls.noncommital
            if not string_to_bool(noncommital, choices=["yes", "no"]):
                q_list.append(gen_q)
        score = self._compute_score(q_list)
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
                             context: str,
                             answer: str,
                             prompts_dict, 
                             encoder):
    try:
        answer_relevance_opt = prompts_dict["answer_relevance"]
    except KeyError:
        answer_relevance_opt = optimize_prompt(encoder)
        prompts_dict["answer_relevance"] = answer_relevance_opt
    pred = answer_relevance_opt(
        question=question, answer=answer, context=context)
    return float(pred.score)

# if True:
#     _ = load_dotenv(find_dotenv())

#     gemini = dspy.Google("models/gemini-1.0-pro",
#                          api_key=os.environ["GOOGLE_API_KEY"],
#                          max_output_tokens=1024,
#                          temperature=0.3)
#     dspy.settings.configure(lm=gemini)
#     encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     answer_relevance = optimize_prompt(encoder)

#     # answer_relevance = AnswerRelevance(encoder=encoder)

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
#     """
#     context = [
#         "- In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.",
#         "- This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.",
#         "- The states with the most restrictive abortion laws have the weakest maternal health support, higher maternal death rates, and higher child poverty rates.",
#         "- The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally and the aid it funds.",
#         "- SRR organizations and activists across the world have expressed fear about the ruling laying the groundwork for anti-abortion legislative and policy attacks in other countries.",
#         "- Advocates have also observed the ruling's impact on progressive law reform and the stalling of the adoption and enforcement of abortion guidelines in certain African countries.",
#         "- The ruling has created a chilling effect in international policy spaces, emboldening anti-abortion state and non-state actors to undermine human rights protections."
#     ]
#     context_str = list_to_string(context, style="number")

#     pred = answer_relevance(
#         question=question,
#         answer=answer,
#         context=context_str)
#     print(pred)






# # # DATA_DIR = "../../data"
# # # DATASET_DIR = os.path.join(DATA_DIR, "dspy-datasets")
# # # RESOURCE_DIR = "../../resources"
# # # CONFIGS_DIR = os.path.join(RESOURCE_DIR, "configs")
# # # CHECKPOINTS_DIR = os.path.join(RESOURCE_DIR, "checkpoints", "answer_relevance_2")

# # # GOLDSET_FP = os.path.join(DATASET_DIR, "answer_relevance.jsonl")
# # # REPORTS_DIR = os.path.join(DATA_DIR, "dspy-reports")
# # # REPORTS_FP = os.path.join(REPORTS_DIR, "answer_relevance.tsv")


# # # class AnswerContextQuestions(dspy.Signature):
# # #     """ Given the answer and context, generate 3 to 5 questions that can be 
# # #         answered by the answer. """
# # #     answer: str = dspy.InputField(desc="the answer")
# # #     context: str = dspy.InputField(desc="the context of answer")
# # #     questions: str = dspy.OutputField(
# # #         desc="list of questions that can be answered by answer")


# # # class QuestionGenerator(dspy.Module):
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.question_generator = dspy.Predict(AnswerContextQuestions)

# # #     def forward(self, answer: str, context: str):
# # #         return self.question_generator(answer=answer, context=context)


# # # # def keywords_match_metric(example, pred, trace=None):
# # # #     A = set([w.lower() for w in nltk.word_tokenize(example.questions)])
# # # #     B = set([w.lower() for w in nltk.word_tokenize(pred.questions)])
# # # #     score = len(A.intersection(B)) / len(A.union(B))
# # # #     if trace is not None:
# # # #         return score
# # # #     else:
# # # #         return score > 0.8


# # # class Similar(dspy.Signature):
# # #     """ Assess if source and target strings are equivalent."""
# # #     source = dspy.InputField(desc="source string")
# # #     target = dspy.InputField(desc="target string")
# # #     is_similar = dspy.OutputField(desc="yes or no")


# # # def question_equivalence_metric(example, pred, trace=None):
# # #     return dspy.Predict(Similar)(
# # #             source=example.questions, target=pred.questions
# # #         ).is_similar.lower() == "yes"


# # # def question_generator_dataset(file_path):
# # #     if not os.path.exists(file_path):
# # #         raise FileNotFoundError(f"answer relevance dataset: {file_path} not found, "
# # #                                 f"create it with generate_datasets.py first.")
# # #     examples = []
# # #     with open(file_path, "r", encoding="utf-8") as fin:
# # #         for i, line in enumerate(fin):
# # #             record = json.loads(line)
# # #             answer = record["answer"]
# # #             context = list_to_string(record["context"], style="number")
# # #             gen_questions = list_to_string(record["gen_questions"],
# # #                                            style="dash")
# # #             examples.append(dspy.Example(
# # #                 answer=answer, context=context, questions=gen_questions
# # #             ).with_inputs("answer", "context"))
# # #     return examples


# # # if True:

# # #     _ = load_dotenv(find_dotenv())
# # #     gemini = dspy.Google("models/gemini-1.0-pro",
# # #                          api_key=os.environ["GOOGLE_API_KEY"],
# # #                          max_output_tokens=1024,
# # #                          temperature=0.3)
# # #     dspy.settings.configure(lm=gemini)

# # #     examples = question_generator_dataset(GOLDSET_FP)
# # #     print(f"#-examples: {len(examples)}")
# # #     trainset, devset = train_test_split(
# # #         examples, test_size=0.3, random_state=42)
# # #     print(f"question generator dataset sizes:"
# # #           f"train={len(trainset)}, dev={len(devset)}")
    
# # #     question_generator = QuestionGenerator()
# # #     teleprompter = BootstrapFewShotWithRandomSearch(
# # #         # metric=keywords_match_metric,
# # #         metric=question_equivalence_metric,
# # #         max_bootstrapped_demos=2,
# # #         max_labeled_demos=2,
# # #         num_threads=1
# # #     )
# # #     question_generator_opt = teleprompter.compile(
# # #         question_generator, trainset=trainset)
    
# # #     ensemble = [prog for *_, prog in
# # #                 question_generator_opt.candidate_programs[:4]]
    
# # #     print("-- eval --")
# # #     evaluate = Evaluate(devset=devset, metric=question_equivalence_metric,
# # #                         num_threads=1, display_progress=True,
# # #                         display_table=1)
# # #     scores = [evaluate(prog) for prog in ensemble]
# # #     print(f"Evaluation scores: {scores}")







# # #     # answer = """
# # #     # "Since you have a penicillin allergy, the treatment options for sinusitis are limited. Some potential options include:

# # #     # - Macrolide antibiotics like azithromycin or clarithromycin. These are usually well tolerated but may not cover all the typical bacteria that cause sinusitis.

# # #     # - Doxycycline may be used, though there is increasing resistance among sinusitis-causing bacteria. 

# # #     # - Respiratory fluoroquinolones like levofloxacin, moxifloxacin or gemifloxacin are effective but should be reserved for infections unresponsive to first-line agents.

# # #     # - Intravenous antibiotics like ceftriaxone or clindamycin may need to be used if the infection is severe or unresponsive to oral agents. 

# # #     # - Nasal corticosteroid sprays may help reduce inflammation and relieve symptoms. Saline irrigation can also help clear mucus and soothe inflamed tissues. 

# # #     # - Analgesics like acetaminophen or ibuprofen may provide pain relief.

# # #     # - Decongestants may help relieve nasal congestion but should be used cautiously.

# # #     # Consult an allergist if the penicillin allergy is questionable. For severe or recurrent cases, referral to an ENT specialist should be considered. Let me know if you have any other questions!"
# # #     # """
# # #     # context = """
# # #     # "Initial drug choice should be based on published data, antibacterial spectrum of activity, and local resistance patterns. Additional patient factors such as allergies, site of infection, renal/hepatic function, and compliance history should also be considered. Therapy may be altered once specific microorganisms are isolated and susceptibilities are known.
# # #     # Spectrum of Activity of Penicillins (PCNs) Against Selected Micro-Organismsa.
# # #     # + = usually clinically effective; +/- = may be clinically effective; blank = not effective or data lacking.
# # #     # aGenerally reported as active; differences exist regionally and locally. Local and patient-specific susceptibility should be applied. Multiple-drug resistance possible for many organisms; bamoxicillin-clavulanic acid; campicillin-sulbactam; dticarcillin-clavulanic acid; epiperacillin-tazobactam."
# # #     # """

# # #     # question_generator = QuestionGenerator()
# # #     # pred = question_generator(answer=answer, context=context)
# # #     # print(pred)
