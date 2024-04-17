from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from xml.sax.saxutils import escape

from .prompt_utils import read_template_from_file, parse_response


PROMPT_CLASSIFY_FACTS = "answer_correctness_1.txt"


def _do_classification(answer: str, ideal_answer: str,
                       model, logger):
    prompt_template = read_template_from_file(PROMPT_CLASSIFY_FACTS)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["answer", "ground_truth"])
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "answer": escape(answer),
        "ground_truth": escape(ideal_answer)
    })
    result = parse_response(response)
    logger.debug("result:", result)
    classification = result.value["classes"]
    return classification


def _get_statements_for_class(statements_dict, class_name):
    try:
        if statements_dict[class_name] is None:
            return []
        else:
            return statements_dict[class_name]["sts"]["st"]
    except KeyError:
        return []


def _compute_answer_correctness_score(statements_by_class_dict):
    tp = len(statements_by_class_dict["TP"])
    fp = len(statements_by_class_dict["FP"])
    fn = len(statements_by_class_dict["FN"])
    score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0.0
    return score


def compute_answer_correctness(answer: str,
                               ideal_answer: str,
                               model: BaseChatModel,
                               logger) -> float:
    classification = _do_classification(answer, ideal_answer, model, logger)
    statements_by_class_dict = {}
    for key in ["TP", "FP", "FN"]:
        statements_by_class_dict[key] = _get_statements_for_class(
            classification, key)
    score = _compute_answer_correctness_score(statements_by_class_dict)
    return score
