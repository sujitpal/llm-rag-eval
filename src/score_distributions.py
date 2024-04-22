import matplotlib.pyplot as plt
import numpy as np
import os


DATA_DIR = "../data"
LCEL_REPORTS_DIR = os.path.join(DATA_DIR, "lcel-reports")
DSPY_REPORTS_DIR = os.path.join(DATA_DIR, "dspy-reports")


def read_data(report_fp):
    scores = []
    with open(report_fp) as f:
        for line in f:
            if line.startswith("#QID"):
                continue
            qid, score = line.strip().split("\t")
            scores.append(float(score))
    return scores


plt.figure(figsize=(10, 5))
for idx, report_fn in enumerate(os.listdir(DSPY_REPORTS_DIR)):
    title = report_fn.split("_report")[0].replace("_", " ").title()
    dspy_report_fp = os.path.join(DSPY_REPORTS_DIR, report_fn)
    lcel_report_fp = os.path.join(LCEL_REPORTS_DIR, report_fn)
    dspy_scores = read_data(dspy_report_fp)
    lcel_scores = read_data(lcel_report_fp)
    print(title, np.std(lcel_scores), np.std(dspy_scores))
    plt.subplot(2, 4, idx + 1)
    plt.hist(lcel_scores, bins=20, alpha=0.5, label="LCEL")
    plt.hist(dspy_scores, bins=20, alpha=0.5, label="DSPy")
    plt.title(title)
    plt.legend(loc="best")

plt.tight_layout()
_ = plt.show()

