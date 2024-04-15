import asyncio
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

from .prompt_utils import read_template_from_file, parse_response


PROMPT_CONTEXT_PRECISION = "context_precision_1.txt"


class Verdict(BaseModel):
    reason: str = Field(alias="reason", description="Reason for verdict")
    infer: str = Field(alias="infer", description="The inference (0/1)")


async def _compute_usefulness_scores(question: str,
                                     context: List[str],
                                     answer: str,
                                     parallel: bool,
                                     model,
                                     logger) -> List[int]:
    
    prompt_template = read_template_from_file(PROMPT_CONTEXT_PRECISION)
    prompt_cprec = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context", "answer"])
    chain_cprec = prompt_cprec | model | StrOutputParser()

    verdicts = []
    if parallel:
        tasks = []
        for ctx in context:
            tasks.append(chain_cprec.ainvoke({
                "question": question,
                "context": ctx,
                "answer": answer
            }))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            result = parse_response(response)
            verdict = Verdict(**result.value["verdict"])
            verdicts.append(verdict)        
    else:
        for ctx in context:
            response = chain_cprec.invoke({
                "question": question,
                "context": ctx,
                "answer": answer
            })
            result = parse_response(response)
            verdicts.append(Verdict(**result.value["verdict"]))

    logger.debug(f"verdicts: {verdicts}")
    scores = [int(verdict.infer) for verdict in verdicts]
    return scores


def _compute_content_precision(scores: List[int]) -> float:
    # precision@k (for k=1..K) discounted by by reciprocal of position
    weights = [sum(scores[:i + 1]) / (i + 1) * scores[i]
               for i in range(len(scores))]
    if len(scores) == 0:
        return 0.0
    context_precision = (
        sum(w * s for w, s in zip(weights, scores)) / len(scores))
    return context_precision


async def compute_context_precision(question: str,
                                    context: List[str],
                                    answer: str,
                                    model: BaseChatModel,
                                    logger,
                                    parallel: bool = True) -> float:

    # prompt_template = read_template_from_file(PROMPT_CONTEXT_PRECISION)
    # prompt_cprec = PromptTemplate(
    #     template=prompt_template,
    #     input_variables=["question", "context", "answer"])
    # chain_cprec = prompt_cprec | model | StrOutputParser()

    # verdicts = []
    # if parallel:
    #     tasks = []
    #     for ctx in context:
    #         tasks.append(chain_cprec.ainvoke({
    #             "question": question,
    #             "context": ctx,
    #             "answer": answer
    #         }))
    #     responses = await asyncio.gather(*tasks)
    #     for response in responses:
    #         result = parse_response(response)
    #         verdict = Verdict(**result.value["verdict"])
    #         verdicts.append(verdict)        
    # else:
    #     for ctx in context:
    #         response = chain_cprec.invoke({
    #             "question": question,
    #             "context": ctx,
    #             "answer": answer
    #         })
    #         result = parse_response(response)
    #         verdicts.append(Verdict(**result.value["verdict"]))

    # logger.debug(f"verdicts: {verdicts}")
    precs = await _compute_usefulness_scores(question, context, answer,
                                             parallel, model, logger)
    context_precision = _compute_content_precision(precs)
    # # precision@k (for k=1..K)
    # scores = [int(verdict.infer) for verdict in verdicts]
    # # weighted by reciprocal of position
    # weights = [sum(scores[:i + 1]) / (i + 1) * scores[i]
    #            for i in range(len(scores))]

    # if len(scores) == 0:
    #     return 0.0
    # context_precision = (
    #     sum(w * s for w, s in zip(weights, scores)) / len(scores))
    return context_precision

