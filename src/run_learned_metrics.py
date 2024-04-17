import argparse
import dspy
import json
import os

from dotenv import find_dotenv, load_dotenv
from enum import Enum
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from learned.faithfulness import compute_faithfulness
from learned.answer_relevance import compute_answer_relevance
from learned.context_precision import compute_context_precision
from learned.context_relevance import compute_context_relevance
from learned.context_recall import compute_context_recall
from learned.answer_correctness import compute_answer_correctness


DATA_DIR = "../data"
REPORTS_DIR = os.path.join(DATA_DIR, "dspy-reports")


class Metrics(Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_UTILIZATION = "context_utilization"
    CONTEXT_RELEVANCE = "context_relevance"
    CONTEXT_RECALL = "context_recall"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


def runner():

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str,
                        choices=sorted([m.value for m in Metrics]),
                        required=True,
                        help="The metric to compute")
    parser.add_argument("--input", type=str, required=True,
                        help="Full path to evaluation data in JSONL format")
    parser.add_argument("--output", type=str, required=False,
                        help="Full path to output TSV file")
    parser.add_argument("--cross-encoder", action="store_false",
                        help="Use cross-encoder similarity scoring (default true)")
    parser.add_argument("--model-temp", type=float, required=False,
                        help="The temperature of the model - between 0.0 and 1.0 (default 0.0)")
    args = parser.parse_args()
    metric = args.metric
    input_fp = args.input
    output_fp = args.output
    if output_fp is None:
        output_fp = os.path.join(REPORTS_DIR, f"{metric}_report.tsv")
    model_temp = args.model_temp
    if model_temp is None or model_temp > 1.0 or model_temp < 0.0:
        model_temp = 0.0

    _ = load_dotenv(find_dotenv())

    model = dspy.Google("models/gemini-1.0-pro",
                        api_key=os.environ["GOOGLE_API_KEY"],
                        max_output_tokens=1024,
                        temperature=0.0)
    dspy.settings.configure(lm=model)

    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    optimized_prompts = {}

    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:

        fout.write("\t".join(["#QID", metric.upper()]) + "\n")
        for line in fin:
            record = json.loads(line)
            # extract relevant data to evaluate
            id = record["id"]
            if int(id) < 19:
                continue
            question = record["query"]
            context = record["context"][0]["chunk_text"][0]
            answer = record["predicted_answer"]
            ideal_answer = record["ideal_answer"]

            match Metrics(metric):
                case Metrics.FAITHFULNESS:
                    metric_value = compute_faithfulness(
                        question, answer, context, optimized_prompts)
                case Metrics.ANSWER_RELEVANCE:
                    metric_value = compute_answer_relevance(
                        question, context, answer, optimized_prompts,
                        encoder)
                case Metrics.CONTEXT_PRECISION:
                    metric_value = compute_context_precision(
                        question, context, ideal_answer, optimized_prompts)
                case Metrics.CONTEXT_UTILIZATION:
                    metric_value = compute_context_precision(
                        question, context, answer, optimized_prompts)
                case Metrics.CONTEXT_RELEVANCE:
                    metric_value = compute_context_relevance(
                        question, context, optimized_prompts)
                case Metrics.CONTEXT_RECALL:
                    context = context.split("\n")
                    metric_value = compute_context_recall(
                        context, answer, optimized_prompts)
                case Metrics.ANSWER_SIMILARITY:
                    raise NotImplementedError(
                        "Use prompted version of answer similarity")
                case Metrics.ANSWER_CORRECTNESS:
                    metric_value = compute_answer_correctness(
                        ideal_answer, answer, optimized_prompts)
                case _:
                    print(f"Unsupported metric: {metric}")

            print(f"query ({id}): {question}, {metric}: {metric_value}")
            fout.write(f"{id}\t{metric_value:.3f}\n")
            # break


if __name__ == "__main__":
    runner()
