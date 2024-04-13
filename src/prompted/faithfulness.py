import asyncio

from itertools import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from typing import List

from .prompt_utils import read_template_from_file, parse_response, Verdict


PROMPT_EXTRACT_STATEMENTS_FROM_ANSWER = "faithfulness_1.txt"
PROMPT_INFER_ENTAILMENT_FROM_CONTEXT = "faithfulness_2.txt"


def _reformat_statements_to_xml(statements: List[str]) -> str:
    statements_xml = ["<statements>"]
    for statement in statements:
        statements_xml.append(f" <statement>{statement}</statement>")
    statements_xml.append("</statements>")
    return "\n".join(statements_xml)


def _get_statements_from_answer(question: str,
                                answer: str,
                                model: BaseChatModel,
                                logger) -> List[str]:
    prompt_template = read_template_from_file(
        PROMPT_EXTRACT_STATEMENTS_FROM_ANSWER)
    prompt_ans_to_stmt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "answer"])
    chain_ans_to_stmt = prompt_ans_to_stmt | model | StrOutputParser()
    response = chain_ans_to_stmt.invoke({
        "question": question,
        "answer": answer
    })
    logger.debug(f"response: {response}")
    result = parse_response(response)
    logger.debug(f"result (parsed response): {result}")
    statements = result.value["statements"]["statement"]
    if not isinstance(statements, list):
        statements = [statements]
        logger.debug("statement made into a []")

    logger.debug(f"statements: {statements}")
    return statements


async def _get_entailments_from_context(context: List[str],
                                        statements: List[str],
                                        model: BaseChatModel,
                                        logger,
                                        parallel: bool
                                        ) -> List[List[float]]:
    statements_xml = _reformat_statements_to_xml(statements)
    logger.debug(f"statements_xml: {statements_xml}")

    prompt_template = read_template_from_file(
        PROMPT_INFER_ENTAILMENT_FROM_CONTEXT)
    prompt_nli = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "statements_xml"])
    chain_nli = prompt_nli | model | StrOutputParser()

    entailments = []
    if parallel:
        tasks = []
        for ctx in context:
            tasks.append(chain_nli.ainvoke({
                "context": ctx,
                "statements_xml": statements_xml
            }))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            result = parse_response(response)
            logger.debug(f"entailment verdicts: {result}")
            verdicts = result.value["verdicts"]["verdict"]
            logger.debug(f"type(verdicts): {type(verdicts)}")
            if not isinstance(verdicts, list):
                verdicts = [verdicts]
                logger.debug(f"made verdicts into []")

            entailments.append([int(Verdict(**v).infer) for v in verdicts])
    else:
        for ctx in context:
            response = chain_nli.invoke({
                "context": ctx,
                "statements_xml": statements_xml
            })
            result = parse_response(response)
            logger.debug(f"entailment verdicts: {result}")
            verdicts = result.value["verdicts"]["verdict"]
            if not isinstance(verdicts, list):
                verdicts = [verdicts]
                logger.debug(f"made verdicts into []")

            entailments.append([int(Verdict(**v).infer) for v in verdicts])

    logger.debug(f"entailments: {entailments}")
    return entailments


def _compute_faithfulness(entailments_lol: List[List[float]]) -> float:
    entailments = list(chain.from_iterable(entailments_lol))
    try:
        faithfulness = sum(entailments) / len(entailments)
    except ZeroDivisionError:
        faithfulness = 0.0
    return faithfulness


async def compute_faithfulness(question: str,
                               answer: str,
                               context: List[str],
                               model: BaseChatModel,
                               logger,
                               parallel: bool = True) -> float:
    statements = _get_statements_from_answer(question, answer, model, logger)
    entailments_lol = await _get_entailments_from_context(
        context, statements, model, logger, parallel)
    faithfulness = _compute_faithfulness(entailments_lol)
    return faithfulness
