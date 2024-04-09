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
                               model: BaseChatModel,
                               logger) -> list:
    more_passages = dsp.retrieve(question, k=3)
    prompt_template = read_template_from_file(PROMPT_MORE_QUESTIONS)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["question", "answer", "passages"])
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "question": question,
        "answer": ideal_answer,
        "passages": more_passages
    })
    result = parse_response(response)
    
    result_pairs = result.value["pairs"]
    pairs = []
    for _ in result_pairs:
        pairs.append((_["question"],_["answer"]))
    logger.debug("pairs: ", pairs)
    # print(f"pairs: {pairs}")
 
    return pairs
