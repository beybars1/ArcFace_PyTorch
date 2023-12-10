import torch, os, time, sys
from models import Backbone
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trans
import torch.nn.functional as F
import copy, random, itertools, cv2
import numpy as np
from sklearn.model_selection import KFold

def create_pairs(id_list, path, k_pairs):
    list_pairs = []
    for id in id_list:
        imgs_list = os.listdir(path+id)
        if len(imgs_list)>1:
            tmp = list(itertools.combinations(imgs_list, 2))
            tmp = [((id+'/'+t[0], id+'/'+t[1]),  True) for t in tmp]
            if len(tmp)>=k_pairs:
                tmp = random.sample(tmp, k_pairs)
            else:
                tmp = random.sample(tmp, len(tmp))
            list_pairs+=tmp
        
        copy_id_list = copy.copy(id_list)
        copy_id_list.remove(id)
        #random_id = random.choice(copy_id_list)
        #random_id_imgs = os.listdir('./cropped_img_celeba/'+random_id)
        tmp2 = []
        for i in range(k_pairs):
            random_id = random.choice(copy_id_list)
            random_id_imgs = os.listdir(path+random_id)
            rch_1 = random.choice(imgs_list)
            rch_2 = random.choice(random_id_imgs)
            tmp2.append(((id+'/'+rch_1, random_id+'/'+rch_2), False))
        list_pairs+=tmp2
    return list_pairs

class DatasetValidation(Dataset):
    def __init__(self, list_pairs, img_dir):
        self.list_pairs = list_pairs
        self.img_dir = img_dir
        self.transform = trans.Compose([trans.ToTensor(),
                                        trans.Normalize(mean=[0.4864, 0.5090, 0.4477], std=[0.2088, 0.1916, 0.1992])])
    #path = '/workspace/face_rec/dataset/celeba/cropped_img_celeba'
    #id_list = os.listdir(path)

    def __len__(self):
        return len(self.list_pairs)

    def __getitem__(self, index):
        img_filename_1 = self.img_dir+self.list_pairs[index][0][0]
        img_filename_2 = self.img_dir+self.list_pairs[index][0][1]
        isSame = self.list_pairs[index][1]

        cv_img_1 = cv2.imread(img_filename_1)
        cv_img_1 = cv2.cvtColor(cv_img_1, cv2.COLOR_BGR2RGB)

        cv_img_2 = cv2.imread(img_filename_2,)
        cv_img_2 = cv2.cvtColor(cv_img_2, cv2.COLOR_BGR2RGB)

        cv_img_1 = self.transform(cv_img_1)
        cv_img_2 = self.transform(cv_img_2)

        output = {
            'IMG_1_NAME':self.list_pairs[index][0][0],
            'IMG_1':cv_img_1,
            'IMG_2_NAME':self.list_pairs[index][0][1],
            'IMG_2':cv_img_2,
            'isSame':isSame
            }
        return output

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
    predict_issame_bool = predict_issame.clone().detach().numpy()

    # print(predict_issame[:5])
    # print()
    # print(actual_issame[:5])
    TP = np.sum(np.logical_and(predict_issame_bool, actual_issame))
    FP = np.sum(np.logical_and(predict_issame_bool, np.logical_not(actual_issame)))
    TN = np.sum(np.logical_and(np.logical_not(predict_issame_bool), np.logical_not(actual_issame)))
    FN = np.sum(np.logical_and(np.logical_not(predict_issame_bool), actual_issame))
    
    # TPR = 0 if (TP + FN == 0) else float(TP) / float(TP + FN)
    # TNR = 0 if (TN + FP == 0) else float(TN) / float(TN + FP)
    # FPR = 0 if (FP + TN == 0) else float(FP) / float(FP + TN)
    # FNR = 0 if (FN + TP == 0) else float(FN) / float(FN + TP)
    acc = float(TP + TN)/dist.shape[0]
    return TP, TN, FP, FN, acc


# def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
#     '''
#     Copy from [insightface](https://github.com/deepinsight/insightface)
#     :param thresholds:
#     :param embeddings1:
#     :param embeddings2:
#     :param actual_issame:
#     :param far_target:
#     :param nrof_folds:
#     :return:
#     '''
#     assert (embeddings1.shape[0] == embeddings2.shape[0])
#     assert (embeddings1.shape[1] == embeddings2.shape[1])
#     nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
#     nrof_thresholds = len(thresholds)
#     k_fold = KFold(n_splits=nrof_folds, shuffle=False)

#     val = np.zeros(nrof_folds)
#     far = np.zeros(nrof_folds)

#     diff = np.subtract(embeddings1, embeddings2)
#     dist = np.sum(np.square(diff), 1)
#     indices = np.arange(nrof_pairs)

#     for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

#         # Find the threshold that gives FAR = far_target
#         far_train = np.zeros(nrof_thresholds)
#         for threshold_idx, threshold in enumerate(thresholds):
#             _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
#         if np.max(far_train) >= far_target:
#             f = interpolate.interp1d(far_train, thresholds, kind='slinear')
#             threshold = f(far_target)
#         else:
#             threshold = 0.0

#         val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

#     val_mean = np.mean(val)
#     far_mean = np.mean(far)
#     val_std = np.std(val)
#     return val_mean, val_std, far_mean


# def calculate_val_far(threshold, dist, actual_issame):
#     predict_issame = np.less(dist, threshold)
#     torch.tensor(a, dtype=torch.bool)
#     true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
#     false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#     n_same = np.sum(actual_issame)
#     n_diff = np.sum(np.logical_not(actual_issame))
#     val = float(true_accept) / float(n_same)
#     far = float(false_accept) / float(n_diff)
#     return val, far


def evaluate_metrics(embeddings_tensor_1, embeddings_tensor_2, isSame, sim_metric, nrof_folds=10):
    thresholds = np.arange(0, 2, 0.01)
    tp, tn, fp, fn, accuracy_test, best_thresholds = calculate_roc(thresholds,\
                                                        embeddings_tensor_1,\
                                                        embeddings_tensor_2,\
                                                        np.asarray(isSame),\
                                                        sim_metric,\
                                                        nrof_folds=nrof_folds
                                                        )
    return tp, tn, fp, fn, accuracy_test, best_thresholds

def evaluate(model, nrof_folds, device, val_dataloader):
    model.eval()
    print("Validation...")
    emb_list_1 = []
    emb_list_2 = []
    isSame_list = []
    with torch.no_grad():
        for val_batch in val_dataloader:
            imgs_1 = val_batch['IMG_1']
            imgs_2 = val_batch['IMG_2']
            isSame = val_batch['isSame']
            imgs_1 = imgs_1.to(device)
            imgs_2 = imgs_2.to(device)
            embeddings_1 = model(imgs_1)
            embeddings_2 = model(imgs_2)
            embeddings_normed_1 = F.normalize(embeddings_1)
            embeddings_normed_2 = F.normalize(embeddings_2)
            embeddings_normed_1 = embeddings_normed_1.detach().cpu()
            embeddings_normed_2 = embeddings_normed_2.detach().cpu()
            emb_list_1.append(embeddings_normed_1)
            emb_list_2.append(embeddings_normed_2)
            isSame_list+=isSame
        embeddings_tensor_1 = torch.vstack(emb_list_1)
        embeddings_tensor_2 = torch.vstack(emb_list_2)
        TPR, TNR, FPR, FNR, accuracy, best_thresholds = evaluate_metrics(embeddings_tensor_1, embeddings_tensor_2,\
                                                                        isSame_list, 'euclidean_dis', nrof_folds)
        # print(TPR)
        # print()
        # print(TNR)
        # print()
        # print(FPR)
        # print()
        # print(FNR)
        # print()
        thresholds = np.arange(0, 2, 0.01)
        for thr in best_thresholds:
            indx=np.where(thresholds == thr)
            ind = indx[0].item()
            #ind = thresholds.index(thr)
            prec = TPR[ind]/(TPR[ind]+FPR[ind])
            rec = TPR[ind]/(TPR[ind]+FNR[ind])
            OF = TNR[ind]/(TNR[ind]+FPR[ind])
            print(prec, rec, OF)
            print()
        print(accuracy)
        print()
        print(best_thresholds)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


if __name__=='__main__':
        
    device = torch.device('cuda:0')
    #init_process_group(backend="nccl")

    model = Backbone(50, 0.6, 'ir_se').to(device)
    #head = Arcface(embedding_size=512, classnum=400000).to(device)
    # model = DDP(model, device_ids=[device])
    # head = DDP(head, device_ids=[device])

    snapshot = torch.load('/data/dev_5/beybars/face_rec/personal_arcFace/distributed_appr/workspace/steps_models_save/model_2023-01-24-05-22_accuracy:0.9887749999999998_epoch:0_step:91980_None.pth', map_location=device)         
    # if device == 0:
    #     print(snapshot.keys())
    # #     print(snapshot)
    # from collections import OrderedDict
    # new_state_dict_model = OrderedDict()
    # new_state_dict_head = OrderedDict()

    # for k, v in snapshot['MODEL_STATE'].items():
    #     if 'module' not in k:
    #         k = 'module.'+k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict_model[k]=v
        
    # for k, v in snapshot["HEAD_STATE"].items():
    #     if 'module' not in k:
    #         k = 'module.'+k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict_head[k]=v

    model.load_state_dict(snapshot['MODEL_STATE'])
    #head.load_state_dict(snapshot["HEAD_STATE"])

    celeba_val='/data/beybars_id/img_data/cropped_alligned_val/'
    id_list = os.listdir(celeba_val)
    list_pairs = create_pairs(id_list, celeba_val, 3)
    val_dataset = DatasetValidation(list_pairs, celeba_val)
    print('here')
    val_dataloader = DataLoader(val_dataset, 64, shuffle=False, num_workers=2)

    accuracy, best_threshold, roc_curve_tensor = evaluate(model=model, nrof_folds=10, device=device, val_dataloader=val_dataloader)