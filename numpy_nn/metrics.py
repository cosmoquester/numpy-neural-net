import numpy as np


def accuracy(label, prediction):
    """
    Calculate accuracy
    :param label: (N, ) Correct label with 0 (negative) or 1 (positive)
    :param prediction: (N, ) Predicted score between 0 and 1
    :returns: (float) Computed accuracy score
    """
    accuracy_score = np.average(np.int32(label == prediction))
    return accuracy_score


def precision(label, prediction):
    """
    Calculate precision
    :param label: (N, ) Correct label with 0 (negative) or 1 (positive)
    :param prediction: (N, ) Predicted score between 0 and 1
    :returns: (float) Computed precision score
    """
    precision_score = np.count_nonzero(np.bitwise_and(label, prediction)) / np.count_nonzero(prediction)
    return precision_score


def recall(label, prediction):
    """
    Calculate recall
    :param label: (N, ) Correct label with 0 (negative) or 1 (positive)
    :param prediction: (N, ) Predicted score between 0 and 1
    :returns: (float) Computed recall score
    """
    recall_socre = np.count_nonzero(np.bitwise_and(label, prediction)) / np.count_nonzero(label)
    return recall_socre


def f1_measure(label, prediction):
    """
    Calculate F-measure
    :param label: (N, ) Correct label with 0 (negative) or 1 (positive)
    :param prediction: (N, ) Predicted score between 0 and 1
    :returns: (float) Computed F-measure score
    """
    precision_score = np.count_nonzero(np.bitwise_and(label, prediction)) / np.count_nonzero(prediction)
    recall_score = np.count_nonzero(np.bitwise_and(label, prediction)) / np.count_nonzero(label)

    if precision_score + recall_score == 0.0:
        return 0.0

    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
    return f1_score


def mean_average_precision(label, hypo, at=10):
    """
    Calculate mAP (mean Average Precision)
    :param label: (N, K), Correct label with 0 (incorrect) or 1 (correct)
    :param hypo: (N, K), Predicted score between 0 and 1
    :param at: (int) # of element to consider from the first. (TOP-@)
    :returns: (float) Computed mAP score
    """
    ap = []
    for i in range(label.shape[0]):
        n_of_relvs = np.count_nonzero(label[i])
        label_p = [x[1] for x in sorted(list(zip(hypo[i], label[i])), key=lambda x: x[0], reverse=True)[:at]]
        Precisions = [np.average(label_p[: i + 1]) for i in range(len(label_p)) if label_p[i]]
        ap.append(sum(Precisions) / n_of_relvs)
    map_score = np.average(ap)

    return map_score


def n_discounted_cumulative_gain(label, hypo, at=10):
    """
    Calculate nDCG
    :param label: (N, K), Correct label with 0 (incorrect) or 1 (correct)
    :param hypo: (N, K), Predicted score between 0 and 1
    :param at: (int) # of element to consider from the first. (TOP-@)
    :returns: (float) Computed nDCG score
    """

    def dcg(label, hypo, at=10):
        label_p = [x[1] for x in sorted(list(zip(hypo, label)), key=lambda x: x[0], reverse=True)[:at]]
        dcg_score = np.sum([1 / np.log2(x + 2) for x in range(len(label_p)) if label_p[x]])
        return dcg_score

    def idcg(label, hypo, at=10):
        n = min(at, np.count_nonzero(label))
        dcgs = [1 / np.log2(x + 2) for x in range(n)]
        idcg_score = np.sum(dcgs)
        return idcg_score

    ndcgs = [dcg(label[i], hypo[i], at) / idcg(label[i], hypo[i], at) for i in range(label.shape[0])]
    ndcg_score = np.average(ndcgs)

    return ndcg_score
