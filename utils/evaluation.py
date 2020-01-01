from sklearn import metrics
import numpy as np


def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / np.sum(dataMat) ** 2
    OA = float(P0 / np.sum(dataMat) * 1.0)
    cohens_coefficient = float((OA - Pe) / (1 - Pe))
    return cohens_coefficient


def compute(y_pred, y_true):
    """
    Do some metrics evaluation.
    :param y_pred: predict label
    :param y_true: ground truth
        example:
        y_pred = [1,2,3,4,5,1,2,1,1,4,5]
        y_true = [1,2,3,4,5,1,2,1,1,4,1]

    :return: OA
    """
    classify_report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    kappa_coefficient = kappa(confusion_matrix, 5)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('kappa coefficient: {0:f}'.format(kappa_coefficient))

    return overall_accuracy


def compute_IoU(y_pred, y_true):
    """
    Evaluate IoU metrics, i.e. Jaccard index.
    Ref:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score
    :param y_pred:
    :param y_true:
    :return:
    """
    IoU = metrics.jaccard_score(y_true, y_pred, average=None)
    mIoU = metrics.jaccard_score(y_true, y_pred, average='macro')
    fwIoU = metrics.jaccard_score(y_true, y_pred, average='weighted')
    print('IoU : \n', IoU)
    print('mIoU: {0:f}'.format(mIoU))
    print('fwIoU: {0:f}'.format(fwIoU))
