import asyncio
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

from common_utils import read_template_from_file, parse_response


PROMPT_CONTEXT_PRECISION = "context_precision_1.txt"


class Verdict(BaseModel):
    reason: str = Field(alias="reason", description="Reason for verdict")
    infer: str = Field(alias="infer", description="The inference (0/1)")


async def compute_context_precision(question: str,
                                    context: List[str],
                                    answer: str,
                                    model: BaseChatModel,
                                    logger,
                                    parallel: bool = True) -> float:

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

    # precision@k (for k=1..K)
    scores = [int(verdict.infer) for verdict in verdicts]
    # weighted by reciprocal of position
    weights = [sum(scores[:i + 1]) / (i + 1) * scores[i]
               for i in range(len(scores))]

    if len(scores) == 0:
        return 0.0
    context_precision = (
        sum(w * s for w, s in zip(weights, scores)) / len(scores))
    return context_precision

