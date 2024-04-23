import matplotlib.pyplot as plt
import numpy as np
import os


REPORTS_DIR = "../resources/reports"
LCEL_REPORTS_DIR = os.path.join(REPORTS_DIR, "lcel-reports")
DSPY_REPORTS_DIR = os.path.join(REPORTS_DIR, "dspy-reports")


def read_data(report_fp):
    scores = []
    with open(report_fp) as f:
        for line in f:
            if line.startswith("#QID"):
                continue
            qid, score = line.strip().split("\t")
            scores.append(float(score))
    return scores


def bimodality(scores):
    """ Measure the "width" of the distribution around 0 and 1.
        All metrics return a value in the range [0, 1]. We measure
        deviation from the mean and then compute the standard deviation
        of the deviations. Smaller values of standard deviation indicate
        a more confident model.
    """
    deviations = [abs(x - 0.5) for x in scores]
    return np.std(deviations)


if __name__ == "__main__":

    print("| Metric | LCEL | DSPy |")
    print("|--------|------|------|")

    plt.figure(figsize=(10, 5))
    for idx, report_fn in enumerate(os.listdir(DSPY_REPORTS_DIR)):
        title = report_fn.split("_report")[0].replace("_", " ").title()
        dspy_report_fp = os.path.join(DSPY_REPORTS_DIR, report_fn)
        lcel_report_fp = os.path.join(LCEL_REPORTS_DIR, report_fn)
        dspy_scores = read_data(dspy_report_fp)
        lcel_scores = read_data(lcel_report_fp)
        lcel_bim = bimodality(lcel_scores)
        dspy_bim = bimodality(dspy_scores)
        print(f"| {title} | {lcel_bim:.3f} | {dspy_bim:.3f} |")
        plt.subplot(2, 4, idx + 1)
        plt.hist(lcel_scores, bins=10, alpha=0.5, label="LCEL")
        plt.hist(dspy_scores, bins=10, alpha=0.5, label="DSPy")
        plt.title(title)
        if idx == 0:
            plt.legend(loc="best")

    plt.tight_layout()
    _ = plt.show()
