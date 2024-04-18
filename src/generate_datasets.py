import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
)
from typing import List

import prompted.faithfulness as faithfulness_p
import prompted.answer_relevance as answer_relevance_p
import prompted.context_precision as context_precision_p
import prompted.context_relevance as context_relevance_p
import prompted.context_recall as context_recall_p
import prompted.answer_correctness as answer_correctness_p
from metrics import Metrics


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


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
    score = faithfulness_p._compute_faithfulness(entailments)
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
    context_str = answer_relevance_p._flatten_context(context)
    gen_questions = \
        answer_relevance_p._generate_questions_from_answer_and_context(
            context_str, answer, 5, model, logger)
    qa_pairs = await answer_relevance_p._predict_noncommittal_from_questions(
        gen_questions, context_str, run_parallel, model,
        logger)
    score = answer_relevance_p._compute_answer_relevance(
        question, qa_pairs, encoder, logger)
    fout.write(json.dumps({
        "id": id,
        "question": question,
        "context": context,
        "answer": answer,
        "gen_questions": gen_questions,
        "non_commitals": [qap.noncommittal for qap in qa_pairs],
        "score": score
    }) + "\n")


async def generate_context_precision_dataset(id: int,
                                             question: str,
                                             answer: str,
                                             context: List[str],
                                             run_parallel: bool,
                                             model,
                                             logger,
                                             fout):
    precs = await context_precision_p._compute_usefulness_scores(
        question, context, answer, run_parallel, model, logger)
    score = context_precision_p._compute_content_precision(precs)
    fout.write(json.dumps({
        "id": id,
        "question": question,
        "context": context,
        "answer": answer,
        "precision": precs,
        "score": score
    }) + "\n")


async def generate_context_relevance_dataset(id: int,
                                             question: str,
                                             context: List[str],
                                             run_parallel: bool,
                                             model,
                                             logger,
                                             fout):
    num_total_sents, context_markdowns = \
        context_relevance_p._convert_to_markdown_lists(context)
    score = 0.0
    if num_total_sents > 0:
        necessary_sents = \
            await context_relevance_p._generate_necessity_verdicts(
                question, context_markdowns, run_parallel, model, logger)
    score = context_relevance_p._compute_context_relevance_score(
        num_total_sents, necessary_sents)
    fout.write(json.dumps({
        "id": id,
        "question": question,
        "context": context,
        "context_sents": context_markdowns,
        "necessary_sents": necessary_sents,
        "score": score
    }) + "\n")


async def generate_context_recall_dataset(id: int,
                                          context: List[str],
                                          answer: str,
                                          run_parallel: bool,
                                          model,
                                          logger,
                                          fout):
    answer_md = context_recall_p._convert_answer_to_markdown_list(
        answer, logger)
    inferences = await \
        context_recall_p._classify_ans_sents_attributable_to_context(
            answer_md, context, run_parallel, model, logger)
    score = context_recall_p._compute_context_recall_score(inferences)
    fout.write(json.dumps({
        "id": id,
        "context": context,
        "answer_md": answer_md,
        "answer": answer,
        "inferences": inferences,
        "score": score
    }) + "\n")


async def generate_answer_correctness_dataset(id: int,
                                              answer: str,
                                              ideal_answer: str,
                                              model,
                                              logger,
                                              fout):
    classification = answer_correctness_p._do_classification(
        answer, ideal_answer, model, logger)
    statements_by_class_dict = {}
    for key in ["TP", "FP", "FN"]:
        statements_by_class_dict[key] = \
            answer_correctness_p._get_statements_for_class(
                classification, key)
    score = answer_correctness_p._compute_answer_correctness_score(
        statements_by_class_dict)
    fout.write(json.dumps({
        "id": id,
        "answer": answer,
        "ideal_answer": ideal_answer,
        "classification": statements_by_class_dict,
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
            # if int(id) != 14:
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
                case Metrics.CONTEXT_PRECISION:
                    await generate_context_precision_dataset(
                        id, question, answer, context, run_parallel,
                        model, logger, fout)
                case Metrics.CONTEXT_RELEVANCE:
                    await generate_context_relevance_dataset(
                        id, question, context, run_parallel, model,
                        logger, fout)
                case Metrics.CONTEXT_RECALL:
                    await generate_context_recall_dataset(
                        id, context, answer, run_parallel, model,
                        logger, fout)
                case Metrics.ANSWER_SIMILARITY:
                    raise NotImplementedError(
                        "Use prompted version of answer similarity")
                case Metrics.ANSWER_CORRECTNESS:
                    await generate_answer_correctness_dataset(
                        id, answer, ideal_answer, model, logger, fout)
                case _:
                    pass


if __name__ == "__main__":
    asyncio.run(runner())
