import asyncio
import nltk

from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from typing import List, Tuple

from .prompt_utils import (
    read_template_from_file, parse_response,
    parse_verdicts_from_result
)


PROMPT_CLASSIFY_NECESSARY = "context_relevance_1.txt"


def _convert_to_markdown_list(context: str) -> Tuple[int, str]:
    context_sents = []
    for sent in nltk.sent_tokenize(context):
        context_sents.append(sent)
    context_markdown = "\n".join([f"- {sent}" for sent in context_sents])
    return len(context_sents), context_markdown


def _convert_to_markdown_lists(context: str) -> Tuple[int, str]:
    total_sents = 0
    context_markdowns = []
    for ctx in context:
        num_sents, context_md = _convert_to_markdown_list(ctx)
        total_sents += num_sents
        context_markdowns.append(context_md)
    return total_sents, context_markdowns


async def _generate_necessity_verdicts(question: str,
                                       context_markdowns: List[str],
                                       parallel: bool,
                                       model,
                                       logger):
    prompt_template = read_template_from_file(PROMPT_CLASSIFY_NECESSARY)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["question", "context"])
    chain = prompt | model | StrOutputParser()

    necessary_sents = []
    if parallel:
        tasks = []
        for context_markdown in context_markdowns:
            tasks.append(chain.ainvoke({
                "question": question,
                "context": context_markdown
            }))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            result = parse_response(response)
            verdicts = parse_verdicts_from_result(result)
            necessary_sents.append([verdict.statement for verdict in verdicts
                                    if int(verdict.infer) == 1])
    else:
        for context_markdown in context_markdowns:
            response = chain.invoke({
                "question": question,
                "context": context_markdown
            })
            result = parse_response(response)
            logger.debug(f"result: {result}")
            verdicts = parse_verdicts_from_result(result)
            necessary_sents.append([verdict.statement for verdict in verdicts
                                    if int(verdict.infer) == 1])
    return necessary_sents


def _compute_context_relevance_score(num_total_sents: int,
                                     necessary_sents: List[List[str]]
                                     ) -> float:
    num_necessary_sents = sum([len(ctx_sents) for ctx_sents in necessary_sents])
    return num_necessary_sents / num_total_sents


async def compute_context_relevance(question: str,
                                    context: List[str],
                                    model: BaseChatModel,
                                    logger,
                                    parallel: bool = True) -> float:

    num_total_sents, context_markdowns = _convert_to_markdown_lists(context)
    score = 0.0
    if num_total_sents > 0:
        necessary_sents = await _generate_necessity_verdicts(
            question, context_markdowns, parallel, model, logger)
        score = _compute_context_relevance_score(
            num_total_sents, necessary_sents)
    return score
