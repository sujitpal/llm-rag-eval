import argparse
import dspy
import json
import logging
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
from learned.learning_utils import clean_up_log_files


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
    parser.add_argument("--output", type=str, required=True,
                        help="Full path to output directory")
    parser.add_argument("--cross-encoder", action="store_false",
                        help="Use cross-encoder similarity scoring (default true)")
    parser.add_argument("--model-temp", type=float, required=False,
                        help="The temperature of the model - between 0.0 and 1.0 (default 0.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Turn debugging on (default: false)")

    args = parser.parse_args()
    metric = args.metric
    input_fp = args.input
    output_dir = args.output
    model_temp = args.model_temp
    if model_temp is None or model_temp > 1.0 or model_temp < 0.0:
        model_temp = 0.0
    debug = args.debug

    _ = load_dotenv(find_dotenv())

    model = dspy.Google("models/gemini-1.0-pro",
                        api_key=os.environ["GOOGLE_API_KEY"],
                        max_output_tokens=1024,
                        temperature=model_temp)
    dspy.settings.configure(lm=model)
    dspy.logger.level = logging.DEBUG if debug else logging.INFO

    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    os.makedirs(output_dir, exist_ok=True)
    output_fp = os.path.join(output_dir, f"{metric}_report.tsv")

    optimized_prompts = {}

    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:

        fout.write("\t".join(["#QID", metric.upper()]) + "\n")
        for line in fin:
            record = json.loads(line)
            # extract relevant data to evaluate
            id = record["id"]
            if int(id) != 14:
                continue
            question = record["query"]
            context = [ctx["chunk_text"] for ctx in record["context"]]
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
                        question, ideal_answer, context, optimized_prompts)
                case Metrics.CONTEXT_UTILIZATION:
                    metric_value = compute_context_precision(
                        question, answer, context, optimized_prompts)
                case Metrics.CONTEXT_RELEVANCE:
                    metric_value = compute_context_relevance(
                        question, context, optimized_prompts)
                case Metrics.CONTEXT_RECALL:
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

    if not debug:
        clean_up_log_files()

if __name__ == "__main__":
    runner()
