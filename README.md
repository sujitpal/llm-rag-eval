# llm-rag-eval

Large Language Model (LLM) powered evaluator for Retrieval Augmented Generation (RAG) pipelines.

<img src="figs/ragas-metrics.drawio.png"/>

## Inspiration

The project is inspired by the [RAGAS project](https://github.com/explodinggradients/ragas) which provides various prompt based approaches to compute various metrics to evaluate a RAG pipeline, and by ideas in the [ARES paper](https://arxiv.org/abs/2311.09476), which attempts to calibrate these LLM evaluators against human evaluators.

## What it does

It provides implementations of metrics that can be applied to outputs of a RAG pipeline in a zero-shot manner, and provides functionality to fine-tune models to generate these metrics targeted to one's use case using small amounts of human annotations.

## How we built it

We used the [DSPy](https://github.com/stanfordnlp/dspy) framework from the ground up to replicate the metrics provided by RAGAS, then extended these metrics to allow for fine-tuning small models using small amounts of human generated evaluations.

## Challenges we ran into

TBD

## Accomplishments that we're proud of

TBD

## What we learned

TBD

## What's next for llm-rag-eval

TBD

