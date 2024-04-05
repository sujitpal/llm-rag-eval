import argparse
import asyncio
import json
import logging
import os

from dotenv import find_dotenv, load_dotenv
from enum import Enum
from langchain_google_genai import (
    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
)

from prompted.faithfulness import compute_faithfulness
from prompted.answer_relevance import compute_answer_relevance
from prompted.context_precision import compute_context_precision
from prompted.context_relevance import compute_context_relevance
from prompted.context_recall import compute_context_recall
from prompted.answer_similarity import compute_answer_similarity
from prompted.answer_correctness import compute_answer_correctness


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


DATA_DIR = "../data"
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


class Metrics(Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_UTILIZATION = "context_utilization"
    CONTEXT_RELEVANCE = "context_relevance"
    CONTEXT_RECALL = "context_recall"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


async def runner():

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str,
                        choices=sorted([m.value for m in Metrics]),
                        required=True,
                        help="The metric to compute")
    parser.add_argument("--input-jsonl", type=str, required=True,
                        help="Full path to evaluation data in JSONL format")
    parser.add_argument("--output-tsv", type=str, required=False,
                        help="Full path to output TSV file")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel where possible (default false)")
    parser.add_argument("--cross-encoder", action="store_false",
                        help="Use cross-encoder similarity scoring (default true)")
    parser.add_argument("--debug", action="store_true",
                        help="Turn debugging on (default: false)")
    args = parser.parse_args()
    metric = args.metric
    input_fp = args.input_jsonl
    output_fp = args.output_tsv
    if output_fp is None:
        output_fp = os.path.join(REPORTS_DIR, f"{metric}_report.tsv")
    run_in_parallel = args.parallel
    use_cross_encoder = args.cross_encoder
    debug = args.debug

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    _ = load_dotenv(find_dotenv())

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.0)
    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:

        fout.write("\t".join(["#QID", metric.upper()]) + "\n")
        for line in fin:
            record = json.loads(line)
            # extract relevant data to evaluate
            id = record["id"]
            if int(id) <= 2:
                continue
            question = record["query"]
            context = [ctx["chunk_text"] for ctx in record["context"]]
            answer = record["predicted_answer"]
            ideal_answer = record["ideal_answer"]

            match Metrics(metric):
                case Metrics.FAITHFULNESS:
                    metric_value = await compute_faithfulness(
                        question, answer, context, model, logger,
                        parallel=run_in_parallel)
                case Metrics.ANSWER_RELEVANCE:
                    metric_value = await compute_answer_relevance(
                        question, context, answer, model, encoder, logger,
                        parallel=run_in_parallel)
                case Metrics.CONTEXT_PRECISION:
                    metric_value = await compute_context_precision(
                        question, context, ideal_answer, model, logger,
                        parallel=run_in_parallel)
                case Metrics.CONTEXT_UTILIZATION:
                    metric_value = await compute_context_precision(
                        question, context, answer, model, logger,
                        parallel=run_in_parallel)
                case Metrics.CONTEXT_RELEVANCE:
                    metric_value = await compute_context_relevance(
                        question, context, model, logger,
                        parallel=run_in_parallel)
                case Metrics.CONTEXT_RECALL:
                    metric_value = await compute_context_recall(
                        context, ideal_answer, model, logger,
                        parallel=run_in_parallel,
                        cross_encoder=use_cross_encoder)
                case Metrics.ANSWER_SIMILARITY:
                    metric_value = compute_answer_similarity(
                        answer, ideal_answer, encoder, logger)
                case Metrics.ANSWER_CORRECTNESS:
                    metric_value = compute_answer_correctness(
                        ideal_answer, answer, model, logger)
                case _:
                    logger.error(f"Unsupported metric: {metric}")

            logger.info(
                f"query ({id}): {question}, {metric}: {metric_value}")
            fout.write(f"{id}\t{metric_value:.3f}\n")
            # break


if __name__ == "__main__":
    asyncio.run(runner())
