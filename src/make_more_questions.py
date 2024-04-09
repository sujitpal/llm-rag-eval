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

from prompted.more_questions import compute_more_questions

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


DATA_DIR = "../data"
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


async def runner():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--metric", type=str,
    #                     choices=sorted([m.value for m in Metrics]),
    #                     required=True,
    #                     help="The metric to compute")
    parser.add_argument("--input-jsonl", type=str, required=True,
                        help="Full path to evaluation data in JSONL format")
    parser.add_argument("--output-tsv", type=str, required=False,
                        help="Full path to output TSV file")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel where possible (default false)")
    # parser.add_argument("--cross-encoder", action="store_false",
    #                     help="Use cross-encoder similarity scoring (default true)")
    parser.add_argument("--debug", action="store_true",
                        help="Turn debugging on (default: false)")
    args = parser.parse_args()
    # metric = args.metric
    input_fp = args.input_jsonl
    output_fp = args.output_tsv
    if output_fp is None:
        output_fp = os.path.join(REPORTS_DIR, f"default_report.tsv")
    run_in_parallel = args.parallel
    # use_cross_encoder = args.cross_encoder
    debug = args.debug

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    _ = load_dotenv(find_dotenv())

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.0)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:

        # fout.write("\t".join(["#QID", metric.upper()]) + "\n")
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
            
            qa_pairs = compute_more_questions(
                question, ideal_answer, model, logger)

            # outputs
            
            o = f"\nquery (id - {id})\nq: {question}"
            logger.info(o)
            print(o)
            fout.write(o)
            o = f"a: {ideal_answer}\n"
            logger.info(o)
            print(o)
            fout.write(o)

            for i, p in enumerate(qa_pairs):
                o = f"q[{i}]: {p[0]}"
                logger.info(o)
                print(o)
                fout.write(o)
                o = f"a[{i}]: {p[1]}"
                logger.info(o)
                print(o)
                fout.write(o)
            

if __name__ == "__main__":
    asyncio.run(runner())
