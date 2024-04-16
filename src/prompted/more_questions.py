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

def split_newlines(input_string):
    """Split the input string at newline characters and return a list of strings."""
    if '\n' in input_string:
        return input_string.split('\n')
    else:
        return [input_string]

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
        chunks = []
        context_lines = split_newlines(_["context"])
        for i in range(len(context_lines)):
            chunk_dict = {}
            chunk_dict['id'] = str(i)
            chunk_dict['chunk_text'] = context_lines[i]
            chunks.append(chunk_dict)
        r_dict = {"query": _["question"], "predicted_answer": _["answer"], "ideal_answer": _["answer"], "context": chunks}
        result_list.append(r_dict)
    logger.debug(f"result_list: \n{result_list}")
 
    return result_list
