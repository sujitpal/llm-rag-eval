import asyncio
import numpy as np

from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

from .prompt_utils import read_template_from_file, parse_response


PROMPT_GEN_QUESTIONS = "answer_relevance_1.txt"
PROMPT_CLASSIFY_NONCOMMITTAL = "answer_relevance_2.txt"


class ClassifiedQAPair(BaseModel):
    question: str = Field(alias="question", description="Generated question")
    answer: str = Field(alias="answer", description="answer to generated question")
    noncommittal: str = Field(alias="noncommittal", description="Noncommittal (0/1)")


def _cosine_similarity(query_vector, doc_vectors):
    query_vector = query_vector.reshape(1, -1)
    sims = np.dot(query_vector, doc_vectors.T) / (
        np.linalg.norm(query_vector) * np.linalg.norm(doc_vectors, axis=1)
    )
    return np.mean(sims)


def _flatten_context(context: List[str]) -> str:
    return " ".join([f"{i+1}. {chunk}" for i, chunk in enumerate(context)])


def _generate_questions_from_answer_and_context(context_flat: str,
                                                answer: str,
                                                num_questions_to_generate: int,
                                                model: BaseChatModel,
                                                logger):
    # generate questions
    prompt_genq = read_template_from_file(PROMPT_GEN_QUESTIONS)
    prompt = PromptTemplate(template=prompt_genq,
                            input_variables=[
                                "num_questions_to_generate",
                                "answer", "context"])
    chain_genq = prompt | model | StrOutputParser()
    response = chain_genq.invoke({
        "num_questions_to_generate": num_questions_to_generate,
        "answer": answer,
        "context": context_flat
    })
    result = parse_response(response)
    gen_questions = result.value["questions"]["question"]
    logger.debug(f"gen_questions: {gen_questions}")
    return gen_questions


async def _predict_noncommittal_from_questions(gen_questions: List[str],
                                               context_flat: str,
                                               parallel: bool,
                                               model: BaseChatModel,
                                               logger):
    # generate answers to generated questions based on context and
    # classify answers as committal or non-committal
    prompt_anc = read_template_from_file(PROMPT_CLASSIFY_NONCOMMITTAL)
    prompt = PromptTemplate(template=prompt_anc,
                            input_variables=["question", "context"])
    chain_anc = prompt | model | StrOutputParser()

    qa_pairs = []
    if parallel:
        tasks = []
        for gen_question in gen_questions:
            tasks.append(chain_anc.ainvoke({
                "question": gen_question,
                "context": context_flat
            }))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            result = parse_response(response)
            qa_pair = ClassifiedQAPair(**result.value["qa_pair"])
            qa_pairs.append(qa_pair)
    else:
        for gen_question in gen_questions:
            response = chain_anc.invoke({
                "question": gen_question,
                "context": context_flat
            })
            result = parse_response(response)
            qa_pair = ClassifiedQAPair(**result.value["qa_pair"])
            qa_pairs.append(qa_pair)

    logger.debug(f"qa_pairs: {qa_pairs}")
    return qa_pairs


def _compute_answer_relevance(question: str,
                              qa_pairs: List[ClassifiedQAPair],
                              encoder: Embeddings,
                              logger):
    # if all non-committal questions, then answer is not relevant
    if np.all([qa_pair.noncommittal == "1" for qa_pair in qa_pairs]):
        logger.warning("cannot compute similarity, generated questions "
                       "are all non-committal")
        return 0.0
    else:
        questions = [question]
        questions.extend([qa_pair.question for qa_pair in qa_pairs])
        embeddings = encoder.embed_documents(questions)
        E = np.array(embeddings)
        source, target = E[0, :], E[1:, :]
        return _cosine_similarity(source, target)


async def compute_answer_relevance(question: str,
                                   context: List[str],
                                   answer: str,
                                   model: BaseChatModel,
                                   encoder: Embeddings,
                                   logger,
                                   num_questions_to_generate: int = 5,
                                   parallel: bool = True
                                   ) -> float:
    # reformat context
    context_flat = _flatten_context(context)
    gen_questions = _generate_questions_from_answer_and_context(
        context_flat, answer, num_questions_to_generate, model, logger)
    qa_pairs = await _predict_noncommittal_from_questions(
        gen_questions, context_flat, parallel, model, logger)
    answer_relevance = _compute_answer_relevance(
        question, qa_pairs, encoder, logger)
    return answer_relevance
