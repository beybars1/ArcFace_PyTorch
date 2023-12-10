import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn, torch
from scipy import interpolate


def calculate_roc(thresholds, embeddings_tensor_1, embeddings_tensor_2, isSame, sim_metric, nrof_folds):
    assert(embeddings_tensor_1.shape[0] == embeddings_tensor_2.shape[0])
    assert(embeddings_tensor_1.shape[1] == embeddings_tensor_2.shape[1])

    nrof_pairs = len(isSame)
    nrof_thresholds = len(thresholds)
    #print(nrof_folds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    tnrs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    fnrs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy_test = np.zeros((nrof_folds))
    accuracy_train = np.zeros((nrof_folds, nrof_thresholds)) #################################################
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if sim_metric=='cosine_sim':
        cos_sim = np.dot(embeddings_tensor_1, embeddings_tensor_2)
    elif sim_metric=='euclidean_dis':
        diff = np.subtract(embeddings_tensor_1, embeddings_tensor_2)
        dist = np.square(diff).sum(axis=1)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, _, _, accuracy_train[fold_idx, threshold_idx] = calculate_accuracy(threshold, dist[train_set], isSame[train_set])
        #best_threshold_index = np.argmax(accuracy_train)
        best_threshold_index = np.argmax(accuracy_train[fold_idx])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], tnrs[fold_idx, threshold_idx],\
            fprs[fold_idx, threshold_idx], fnrs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,\
                                                                                                dist[test_set],\
                                                                                                isSame[test_set])
        _, _, _, _, accuracy_test[fold_idx] = calculate_accuracy(thresholds[best_threshold_index],\
                                                                            dist[test_set],\
                                                                            isSame[test_set])
    tpr = np.mean(tprs, 0)
    tnr = np.mean(tnrs, 0)
    fpr = np.mean(fprs, 0)
    fnr = np.mean(fnrs, 0)
    return tpr, tnr, fpr, fnr, accuracy_test, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    predict_issame_bool = torch.tensor(predict_issame.clone().detach(), dtype=torch.bool).numpy()
    # print(predict_issame[:5])
    # print()
    # print(actual_issame[:5])
    TP = np.sum(np.logical_and(predict_issame_bool, actual_issame))
    FP = np.sum(np.logical_and(predict_issame_bool, np.logical_not(actual_issame)))
    TN = np.sum(np.logical_and(np.logical_not(predict_issame_bool), np.logical_not(actual_issame)))
    FN = np.sum(np.logical_and(np.logical_not(predict_issame_bool), actual_issame))
    
    TPR = 0 if (TP + FN == 0) else float(TP) / float(TP + FN)
    TNR = 0 if (TN + FP == 0) else float(TN) / float(TN + FP)
    FPR = 0 if (FP + TN == 0) else float(FP) / float(FP + TN)
    FNR = 0 if (FN + TP == 0) else float(FN) / float(FN + TP)
    acc = float(TP + TN)/dist.shape[0]
    return TPR, TNR, FPR, FNR, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    '''
    Copy from [insightface](https://github.com/deepinsight/insightface)
    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    torch.tensor(a, dtype=torch.bool)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate_metrics(embeddings_tensor_1, embeddings_tensor_2, isSame, sim_metric, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    tp, tn, fp, fn, accuracy_test, best_thresholds = calculate_roc(thresholds,\
                                                        embeddings_tensor_1,\
                                                        embeddings_tensor_2,\
                                                        np.asarray(isSame),\
                                                        sim_metric,\
                                                        nrof_folds=nrof_folds
                                                        )
    return tp, tn, fp, fn, accuracy_test, best_thresholds