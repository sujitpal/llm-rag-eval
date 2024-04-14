# llm-rag-eval

Large Language Model (LLM) powered evaluator for Retrieval Augmented Generation (RAG) pipelines.

<img src="figs/ragas-metrics.drawio.png"/>

## Google AI Hackathon Submission

### Inspiration

The project is inspired by the [RAGAS project](https://github.com/explodinggradients/ragas) which provides various prompt based approaches to compute various metrics to evaluate a RAG pipeline, and by ideas in the [ARES paper](https://arxiv.org/abs/2311.09476), which attempts to calibrate these LLM evaluators against human evaluators.

### What it does

It provides implementations of metrics that can be applied to outputs of a RAG pipeline in a zero-shot manner, and provides functionality to fine-tune models to generate these metrics targeted to one's use case using small amounts of human annotations.

### How we built it

We used the [DSPy](https://github.com/stanfordnlp/dspy) framework from the ground up to replicate the metrics provided by RAGAS, then extended these metrics to allow for fine-tuning small models using small amounts of human generated evaluations.

### Challenges we ran into

TBD

### Accomplishments that we're proud of

TBD

### What we learned

TBD

### What's next for llm-rag-eval

TBD

## Running Prompted RAGAS metrics

The following RAGAS metrics have been (re-)implemented in this project (because
I had trouble making them work as-is, and because they are conceptually quite 
simple to implement).

* [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html) -- given question, context and answer, measures factual consistency of the answer against the given context.
* [Answer Relevance](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html) -- given question, context and answer, measures how pertinent the answer is to the question.
* [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html) -- given question, context and ground truth, measures whether statements in the ground truth are found in context.
* Context Utilization -- same as Context Precision, using the predicted answer instead of the ground truth.
* [Context Relevance](https://docs.ragas.io/en/stable/concepts/metrics/context_relevancy.html) -- given question and context, measures relevance of the context against the question.
* [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html) -- given the context and answer, measures extent to which context aligns with the answer.
* [Answer Similarity](https://docs.ragas.io/en/stable/concepts/metrics/semantic_similarity.html) -- given answer and ground truth, measures semantic similarity between them (cosine or cross encoder similarity).
* [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html) -- given answer and ground truth, measures the accuracy given facts in these two texts.

The metrics described above can be run against your dataset by calling the `run_prompted_metrics.py` script in the `src` directory, with the path to the input (provided in JSON-L), the path to the output (TSV) file, and the appropriate metric name.

```
$ python3 run_prompted_metrics.py --help
usage: run_prompted_metrics.py [-h] --metric
                               {answer_correctness,answer_relevance,answer_similarity,context_precision,context_recall,context_relevance,context_utilization,faithfulness}
                               --input INPUT_JSONL [--output OUTPUT_TSV] [--parallel] [--cross-encoder]

options:
  -h, --help            show this help message and exit
  --metric {answer_correctness,answer_relevance,answer_similarity,context_precision,context_recall,context_relevance,context_utilization,faithfulness}
                        The metric to compute
  --input INPUT_JSONL
                        Full path to evaluation data in JSONL format
  --output OUTPUT_TSV
                        Full path to output TSV file
  --parallel            Run in parallel where possible (default false)
  --cross-encoder       Use cross-encoder similarity scoring (default false)
```

Ideally, we want to generate metrics from a running RAG pipeline, but in order to simplify the development process, we have isolate the evaluation functionality, feeding it the input it needs via a JSON-L file. Each line of the JSON-L file represents a single RAG transaction. The required fields are as follows.

```
{
    "id": {qid: int},
    "query": {query: str},
    "context": [
        {
            "id": {chunk_id: str},
            "chunk_text": {chunk_text: str},
            ... other relevant fields
        },
        ... more context elements
    ],
    "ideal_answer": {ground_truth: str},
    "predicted_answer": {answer: str}
}
```

We have used the [AmnestyQA](https://huggingface.co/datasets/explodinggradients/amnesty_qa) dataset on HuggingFace as our reference dataset. You can find a copy of that data in the format described above.

## Running Learned RAGAS metrics

We have used DSPy to optimize our prompts for AmnestyQA dataset. At a very high level, this involves using random subsets of the training data (in our case outputs from our prompted RAG metrics) and finding the best subset of examples that produce the most optimized prompt.

The DSPy implementations look for the optimized configuration in `resource/configs`. If it doesn't find it, then it looks for the dataset to allow it to optimize itself first. We have provided configurations for RAGAS metrics prompts optimized for AmnestyQA, but you probably need to generate optimized versions for your own dataset. To do so, you need to run the `generate_datasets.py` script, which will extract data from running the RAGAS prompts against the LLM and write it out as a JSON file into `data/dspy-datasets` (that's where the DSPy fine-tuning code expects to find it). The command to generate a dataset to fine-tune a DSPy prompt for a particular metric is shown below:

```
$ python3 generate_datasets.py --help
usage: generate_datasets.py [-h] --metric
                            {answer_correctness,answer_relevance,answer_similarity,context_precision,context_recall,context_relevance,context_utilization,faithfulness}
                            --input INPUT --output OUTPUT [--parallel] [--debug]

options:
  -h, --help            show this help message and exit
  --metric {answer_correctness,answer_relevance,answer_similarity,context_precision,context_recall,context_relevance,context_utilization,faithfulness}
                        The metric to generate datasets for
  --input INPUT         Full path to input JSONL file
  --output OUTPUT       Full path to output directory
  --parallel            Run in parallel where possible (default false)
  --debug               Turn debugging on (default: false)
```

To re-run the optimization locally, remove the configuration file for the metric from the `resources/config` directory. The next time you run `run_learned_metrics.py` it will re-optimize (this is a fairly lengthy process but doesn't require GPU). Leave the config file alone to re-use the prompt optimized for AmnestyQA.

```
$ python3 run_learned_metrics.py --help
usage: run_learned_metrics.py [-h] --metric
                              {answer_correctness,answer_relevance,answer_similarity,context_precision,context_recall,context_relevance,context_utilization,faithfulness}
                              --input INPUT [--output OUTPUT] [--cross-encoder] [--model-temp MODEL_TEMP]

options:
  -h, --help            show this help message and exit
  --metric {answer_correctness,answer_relevance,answer_similarity,context_precision,context_recall,context_relevance,context_utilization,faithfulness}
                        The metric to compute
  --input INPUT         Full path to evaluation data in JSONL format
  --output OUTPUT       Full path to output TSV file
  --cross-encoder       Use cross-encoder similarity scoring (default true)
  --model-temp MODEL_TEMP
                        The temperature of the model - between 0.0 and 1.0 (default 0.0)
```
