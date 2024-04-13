import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv, find_dotenv
from enum import Enum
from langchain_google_genai import (
    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
)
from typing import List

import prompted.faithfulness as faithfulness_p
import prompted.answer_relevance as answer_relevance_p


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class Metrics(Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_UTILIZATION = "context_utilization"
    CONTEXT_RELEVANCE = "context_relevance"
    CONTEXT_RECALL = "context_recall"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


async def generate_faithfulness_dataset(id: int,
                                        question: str,
                                        answer: str,
                                        context: List[str],
                                        run_parallel: bool,
                                        model,
                                        logger,
                                        fout):
    statements = faithfulness_p._get_statements_from_answer(
        question, answer, model, logger)
    entailments = await faithfulness_p._get_entailments_from_context(
        context, statements, model, logger,
        parallel=run_parallel)
    score = await faithfulness_p.compute_faithfulness(
        question, answer, context, model, logger,
        parallel=run_parallel)
    fout.write(json.dumps({
        "id": id,
        "question": question,
        "context": context,
        "answer": answer,
        "statements": statements,
        "entailments": entailments,
        "score": score
    }) + "\n")


async def generate_answer_relevance_dataset(id: int,
                                            question: str,
                                            context: List[str],
                                            answer: str,
                                            run_parallel: bool,
                                            model,
                                            encoder,
                                            logger,
                                            fout):
    context_flat = answer_relevance_p._flatten_context(context)
    gen_questions = answer_relevance_p._generate_questions_from_answer_and_context(
        context_flat, answer, 5, model, logger)
    qa_pairs = await answer_relevance_p._predict_noncommittal_from_questions(
        gen_questions, context_flat, run_parallel, model,
        logger)
    score = answer_relevance_p._compute_answer_relevance(
        question, qa_pairs, encoder, logger)
    fout.write(json.dumps({
        "id": id,
        "context": context,
        "answer": answer,
        "gen_questions": gen_questions,
        "non_commitals": [qap.noncommittal for qap in qa_pairs],
        "score": score
    }) + "\n")


async def runner():

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str,
                        choices=sorted([m.value for m in Metrics]),
                        required=True,
                        help="The metric to generate datasets for")
    parser.add_argument("--input", type=str, required=True,
                        help="Full path to input JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Full path to output directory")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel where possible (default false)")
    parser.add_argument("--debug", action="store_true",
                        help="Turn debugging on (default: false)")
    args = parser.parse_args()
    metric = args.metric
    input_fp = args.input
    output_fp = os.path.join(args.output, f"{metric}.jsonl")
    run_parallel = args.parallel

    _ = load_dotenv(find_dotenv())

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.0)
    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    os.makedirs(args.output, exist_ok=True)

    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            id = record["id"]
            # if int(id) < 19:
            #     continue
            question = record["query"]
            context = [ctx["chunk_text"] for ctx in record["context"]]
            answer = record["predicted_answer"]
            ideal_answer = record["ideal_answer"]

            logger.info(f"Processing query ({id}): {question}")

            match Metrics(metric):
                case Metrics.FAITHFULNESS:
                    await generate_faithfulness_dataset(
                        id, question, answer, context, run_parallel,
                        model, logger, fout)
                case Metrics.ANSWER_RELEVANCE:
                    await generate_answer_relevance_dataset(
                        id, question, context, answer, run_parallel,
                        model, encoder, logger, fout)
                case _:
                    pass

            # break


if __name__ == "__main__":
    asyncio.run(runner())