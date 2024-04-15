import asyncio
import nltk

from itertools import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from typing import List
from xml.sax.saxutils import escape

from .prompt_utils import (
    read_template_from_file, parse_response,
    parse_verdicts_from_result
)

PROMPT_CLASSIFY_ATTRIB = "context_recall_1.txt"


def _convert_answer_to_markdown_list(answer: str,
                                     logger,
                                     max_sents: int = 10) -> str:
    answer_sents = []
    for sent in nltk.sent_tokenize(answer):
        sent = escape(sent)
        answer_sents.append(sent)
    answer_markdown = "\n".join([f"- {sent}" for sent in answer_sents])
    # to deal with context length limitations (this hack is to take
    # care of lines that are already in the list format before sent_tokenize
    answer_sents = answer_markdown.split("\n- ")[:max_sents]
    answer_markdown = "\n- ".join(answer_sents)
    logger.debug(f"answer_md ({len(answer_sents)} sentences): {answer_markdown}")
    return answer_markdown


async def _classify_ans_sents_attributable_to_context(answer_md: str,
                                                      context: List[str],
                                                      parallel: bool,
                                                      model,
                                                      logger) -> List[int]:
    prompt_template = read_template_from_file(PROMPT_CLASSIFY_ATTRIB)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "answer"]
    )
    chain = prompt | model | StrOutputParser()

    inferences = []
    if parallel:
        tasks = []
        for ctx in context:
            tasks.append(chain.invoke({
                "context": ctx,
                "answer": answer_md
            }))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            result = parse_response(response)
            verdicts = parse_verdicts_from_result(result)
            inferences.append([int(verdict.infer) for verdict in verdicts])
    else:
        for ctx in context:
            response = chain.invoke({
                "context": ctx,
                "answer": answer_md
            })
            result = parse_response(response)
            verdicts = parse_verdicts_from_result(result)
            inferences.append([int(verdict.infer) for verdict in verdicts])
    
    logger.debug(f"inferences: {inferences}")
    return inferences


def _compute_context_recall_score(inferences: List[int]) -> float:
    inferences_f = list(chain(*inferences))
    score = 0.0
    if len(inferences_f) > 0:
        score = sum(inferences_f) / len(inferences_f)
    return score


async def compute_context_recall(context: List[str],
                                 answer: str,
                                 model: BaseChatModel,
                                 logger,
                                 parallel: bool = True) -> float:

    answer_md = _convert_answer_to_markdown_list(answer, logger)
    inferences = _classify_ans_sents_attributable_to_context(
        answer_md, context, parallel, model, logger)
    score = _compute_context_recall_score(inferences)
    return score
