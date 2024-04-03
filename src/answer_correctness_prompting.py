from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from xml.sax.saxutils import escape

from common_utils import read_template_from_file, parse_response


PROMPT_CLASSIFY_FACTS = "answer_correctness_1.txt"


def compute_answer_correctness(answer: str,
                               ideal_answer: str,
                               model: BaseChatModel,
                               logger) -> float:
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
    num_facts_by_class = {}
    for key in ["TP", "FP", "FN"]:
        try:
            if classification[key] is None:
                num_facts_by_class[key] = 0
            else:
                facts = classification[key]["sts"]["st"]
                num_facts_by_class[key] = len(facts)
        except KeyError:
            num_facts_by_class[key] = 0
    tp = num_facts_by_class["TP"]
    fp = num_facts_by_class["FP"]
    fn = num_facts_by_class["FN"]
    score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0.0
    return score
