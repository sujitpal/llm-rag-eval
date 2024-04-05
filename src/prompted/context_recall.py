import asyncio
import nltk

from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from typing import List, Tuple
from xml.sax.saxutils import escape

from .prompt_utils import (
    read_template_from_file, parse_response,
    parse_verdicts_from_result
)

PROMPT_CLASSIFY_ATTRIB = "context_recall_1.txt"


def _convert_answer_to_markdown_list(answer: str,
                                     max_sents: int = 10
                                     ) -> Tuple[int, str]:
    answer_sents = []
    for sent in nltk.sent_tokenize(answer):
        sent = escape(sent)
        answer_sents.append(sent)
    answer_markdown = "\n".join([f"- {sent}" for sent in answer_sents])
    # to deal with context length limitations (this hack is to take
    # care of lines that are already in the list format before sent_tokenize
    answer_sents = answer_markdown.split("\n- ")[:max_sents]
    answer_markdown = "\n- ".join(answer_sents)
    return len(answer_sents), answer_markdown


async def compute_context_recall(context: List[str],
                                 answer: str,
                                 model: BaseChatModel,
                                 logger,
                                 parallel: bool = True) -> float:

    prompt_template = read_template_from_file(PROMPT_CLASSIFY_ATTRIB)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "answer"]
    )
    chain = prompt | model | StrOutputParser()

    num_ans_sents, answer_md = _convert_answer_to_markdown_list(answer)
    logger.debug(f"answer_md ({num_ans_sents} sentences): {answer_md}")

    all_verdicts = []
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
            all_verdicts.extend(verdicts)
    else:
        for ctx in context:
            response = chain.invoke({
                "context": ctx,
                "answer": answer_md
            })
            result = parse_response(response)
            verdicts = parse_verdicts_from_result(result)
            all_verdicts.extend(verdicts)
    
    logger.debug(f"all verdicts: {all_verdicts}")

    num_attributable = sum([int(verdict.infer) for verdict in all_verdicts])
    num_denom = len(context) * num_ans_sents
    if num_denom == 0:
        return 0.0
    recall = num_attributable / num_denom
    return recall
