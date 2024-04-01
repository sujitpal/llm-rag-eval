import asyncio
import json
import logging
import numpy as np
import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

from common_utils import read_template_from_file, parse_response


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


PROMPT_GEN_QUESTIONS = "answer_relevance_1.txt"
PROMPT_CLASSIFY_NONCOMMITTAL = "answer_relevance_2.txt"

DATA_DIR = "../data"
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


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


async def compute_answer_relevance(question: str,
                                   context: List[str],
                                   answer: str,
                                   model: ChatGoogleGenerativeAI,
                                   encoder: GoogleGenerativeAIEmbeddings,
                                   logger,
                                   num_questions_to_generate: int = 5
                                   ) -> float:
    # reformat context
    context_flat = " ".join([f"{i+1}. {chunk}" 
                             for i, chunk in enumerate(context)])
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

    # generate answers to generated questions based on context and
    # classify answers as committal or non-committal
    prompt_anc = read_template_from_file(PROMPT_CLASSIFY_NONCOMMITTAL)
    prompt = PromptTemplate(template=prompt_anc,
                            input_variables=["question", "context"])
    chain_anc = prompt | model | StrOutputParser()
    tasks = []
    for gen_question in gen_questions:
        tasks.append(chain_anc.ainvoke({
            "question": gen_question,
            "context": context
        }))
    responses = await asyncio.gather(*tasks)
    qa_pairs = []
    for response in responses:
        result = parse_response(response)
        qa_pair = ClassifiedQAPair(**result.value["qa_pair"])
        qa_pairs.append(qa_pair)
    logger.debug(f"qa_pairs: {qa_pairs}")

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


async def runner():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    _ = load_dotenv(find_dotenv())

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3)
    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    input_fp = os.path.join(DATA_DIR, "goldset_ragas.jsonl")
    output_fp = os.path.join(REPORTS_DIR, "answer_relevance_report.tsv")
    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:
        fout.write("\t".join(["#QID", "ANSWER_RELEVANCE"]) + "\n")
        for line in fin:
            record = json.loads(line)
            id = record["id"]
            question = record["query"]
            context = [ctx["chunk_text"] for ctx in record["context"]]
            answer = record["predicted_answer"]
            relevance = await compute_answer_relevance(
                question, context, answer, model, encoder, logger)
            logger.info(
                f"query ({id}): {question}, answer relevance: {relevance:.3f}")
            fout.write(f"{id}\t{relevance:.3f}\n")
            # break


if __name__ == "__main__":
    asyncio.run(runner())
