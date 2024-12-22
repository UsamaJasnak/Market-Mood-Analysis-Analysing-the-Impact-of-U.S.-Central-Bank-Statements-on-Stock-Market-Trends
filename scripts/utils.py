import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, roc_auc_score


def test_report(Y_test, labels, probs):
    """Create a test performance report.

    Parameters:
        Y_test: test set target values.
        labels: list of predicted labels.
        probs: list of predicted probabilities.
    """

    # Classification report
    report = pd.DataFrame(classification_report(Y_test, labels, output_dict=True))

    # Additional arguments
    accuracy = accuracy_score(Y_test, labels)
    balanced_accuracy = balanced_accuracy_score(Y_test, labels)
    auc = roc_auc_score(Y_test, probs)

    report['accuracy'] = accuracy
    report['balanced accuracy'] = balanced_accuracy
    report['auc'] = auc

    return report