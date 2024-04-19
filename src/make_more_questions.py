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

# Safety config

from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
}

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
    parser.add_argument("--id-start", type=int, required=False,
                        help="The number that the question ids in the output should start with (default 0)")
    parser.add_argument("--multiplier", type=int, required=False,
                        help="The number of new questions to be generated PER question in input data (default 3)")
    parser.add_argument("--model-temp", type=float, required=False,
                        help="The temperature of the model - between 0.0 and 1.0 (default 0.0)")
    parser.add_argument("--max", type=int, required=False,
                        help="The maximum number of new questions to be generated total (no default)")
   
    args = parser.parse_args()
    # metric = args.metric
    input_fp = args.input_jsonl
    output_fp = args.output_tsv
    if output_fp is None:
        output_fp = os.path.join(REPORTS_DIR, f"default_report.tsv")
    run_in_parallel = args.parallel
    # use_cross_encoder = args.cross_encoder
    debug = args.debug
    id_start = args.id_start
    if id_start is None:
        id_start = 0
    multiplier = args.multiplier
    if multiplier is None:
        multiplier = 3
    model_temp = args.model_temp
    if model_temp is None or model_temp > 1.0 or model_temp < 0.0:
        model_temp = 0.0
    maxq = args.max
    if maxq is None:
        maxq = 99999
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    _ = load_dotenv(find_dotenv())

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=model_temp,
        safety_settings=safety_settings)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(input_fp, "r", encoding="utf-8") as fin, \
         open(output_fp, "w", encoding="utf-8") as fout:

        # fout.write("\t".join(["#QID", metric.upper()]) + "\n")
        q_counter = 0
        new_q_id = id_start
        for line in fin:
            record = json.loads(line)
            # extract relevant data to evaluate
            id = record["id"]
            # be done if hit max
            if q_counter + multiplier > maxq:
                break
            question = record["query"]
            context = [ctx["chunk_text"] for ctx in record["context"]]
            answer = record["predicted_answer"]
            ideal_answer = record["ideal_answer"]
            
            more_questions = compute_more_questions(
                question, ideal_answer, multiplier, model, logger)
            
            q_counter = q_counter + multiplier

            for q in more_questions:
                q["id"] = new_q_id
                new_q_id += 1
                fout.write(json.dumps(q) + '\n')


if __name__ == "__main__":
    asyncio.run(runner())
