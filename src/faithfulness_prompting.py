import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from typing import List

from common_utils import read_template_from_file, parse_response, Verdict


PROMPT_EXTRACT_STATEMENTS_FROM_ANSWER = "faithfulness_1.txt"
PROMPT_INFER_ENTAILMENT_FROM_CONTEXT = "faithfulness_2.txt"


def reformat_statements_to_xml(statements: List[str]) -> str:
    statements_xml = ["<statements>"]
    for statement in statements:
        statements_xml.append(f" <statement>{statement}</statement>")
    statements_xml.append("</statements>")
    return "\n".join(statements_xml)


async def compute_faithfulness(question: str,
                               answer: str,
                               context: List[str],
                               model: BaseChatModel,
                               logger,
                               parallel: bool = True) -> float:
    prompt_template = read_template_from_file(
        PROMPT_EXTRACT_STATEMENTS_FROM_ANSWER)
    prompt_ans_to_stmt = PromptTemplate(template=prompt_template,
                                        input_variables=["question", "answer"])
    chain_ans_to_stmt = prompt_ans_to_stmt | model | StrOutputParser()
    response = chain_ans_to_stmt.invoke({
        "question": question,
        "answer": answer
    })
    result = parse_response(response)
    logger.debug(f"extracted statements from answer: {result}")
    statements = result.value["statements"]["statement"]
    
    statements_xml = reformat_statements_to_xml(statements)
    logger.debug(f"statements_xml: {statements_xml}"
                 )
    prompt_template = read_template_from_file(
        PROMPT_INFER_ENTAILMENT_FROM_CONTEXT)
    prompt_nli = PromptTemplate(template=prompt_template,
                                input_variables=["context", "statements_xml"])
    chain_nli = prompt_nli | model | StrOutputParser()

    num_entailed, num_total = 0, 0

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
            for verdict_dict in verdicts:
                verdict = Verdict(**verdict_dict)
                if verdict.infer == "1":
                    num_entailed += 1
                num_total += 1
    else:
        for ctx in context:
            response = chain_nli.invoke({
                "context": ctx,
                "statements_xml": statements_xml
            })
            result = parse_response(response)
            logger.debug(f"entailment verdicts: {result}")
            verdicts = result.value["verdicts"]["verdict"]
            for verdict_dict in verdicts:
                verdict = Verdict(**verdict_dict)
                if verdict.infer == "1":
                    num_entailed += 1
                num_total += 1

    logger.debug(f"num_entailed: {num_entailed}, num_total: {num_total}")

    try:
        faithfulness = num_entailed / num_total
    except ZeroDivisionError:
        faithfulness = 0.0
    return faithfulness
