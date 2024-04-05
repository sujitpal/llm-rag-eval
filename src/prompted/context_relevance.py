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


async def compute_context_relevance(question: str,
                                    context: List[str],
                                    model: BaseChatModel,
                                    logger,
                                    parallel: bool = True) -> float:

    total_sents = 0
    context_markdowns = []
    for ctx in context:
        num_sents, context_md = _convert_to_markdown_list(ctx)
        total_sents += num_sents
        context_markdowns.append(context_md)

    if total_sents == 0:
        return 0.0

    prompt_template = read_template_from_file(PROMPT_CLASSIFY_NECESSARY)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["question", "context"])
    chain = prompt | model | StrOutputParser()

    num_required_sents = 0
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
            num_required_sents += sum([int(verdict.infer) for verdict in verdicts])
    else:
        for context_markdown in context_markdowns:
            response = chain.invoke({
                "question": question,
                "context": context_markdown
            })
            result = parse_response(response)
            logger.debug(f"result: {result}")
            verdicts = parse_verdicts_from_result(result)
            num_required_sents += sum([int(verdict.infer) for verdict in verdicts])

    return num_required_sents / total_sents
