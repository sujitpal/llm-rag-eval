from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from xml.sax.saxutils import escape

import dspy
import dsp

from .prompt_utils import read_template_from_file, parse_response

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

PROMPT_MORE_QUESTIONS = "more_questions.txt"


def compute_more_questions(question: str,
                               ideal_answer: str,
                               multiplier: int,
                               model: BaseChatModel,
                               logger) -> list:
    more_passages = dsp.retrieve(question, k=multiplier)
    esc_more_passages = [escape(x) for x in more_passages]
    prompt_template = read_template_from_file(PROMPT_MORE_QUESTIONS)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["question", "answer", "multiplier", "passages"])
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "question": question,
        "answer": ideal_answer,
        "multiplier": multiplier,
        "passages": esc_more_passages
    })
    
    # make list of dicts from XML output
    # format per requirement
    # ideal_answer is just the predicted_answer, as a workaround

    result = parse_response(response)
    
    result_tuples = result.value["tuples"]
    result_list = []
    for _ in result_tuples:
        chunk = {"id": "1", "chunk_text": _["context"]}
        r_dict = {"query": _["question"], "predicted_answer": _["answer"], "ideal_answer": _["answer"], "context": [chunk]}
        result_list.append(r_dict)
    logger.debug("result_list: ", result_list)
 
    return result_list
