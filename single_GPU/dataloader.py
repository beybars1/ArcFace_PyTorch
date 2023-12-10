from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from torchvision import transforms as trans
import itertools, random, copy, os
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import mxnet as mx
from tqdm import tqdm

# create validation pairs
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
    
# create train list
def create_train_list(path):
    tmp = os.listdir(path)
    list_imgs = []
    labels = []
    for t in tmp:
        tmp_tmp = os.listdir(path+t)
        for tt in tmp_tmp:
            list_imgs.append(t+'/'+tt)
            labels.append(t)
    le = preprocessing.LabelEncoder()
    labels_new = le.fit_transform(labels)
    return list_imgs, labels_new


class DatasetTrain(Dataset):
    def __init__(self, list_imgs, img_dir, labels):
        self.list_imgs = list_imgs
        self.img_dir = img_dir
        self.labels = labels
        self.transform = trans.Compose([trans.ToTensor(),
                                        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        img_filename_1 = self.img_dir+self.list_imgs[index]
        cv_img_1 = cv2.imread(img_filename_1)
        cv_img_1 = cv2.cvtColor(cv_img_1, cv2.COLOR_BGR2RGB)
        cv_img_1 = self.transform(cv_img_1)
        label = self.labels[index]

        output = {
            'IMG_1_NAME':self.list_imgs[index],
            'IMG_1':cv_img_1,
            'LABEL':label
            }
        return output


class DatasetValidation(Dataset):
    def __init__(self, list_pairs, img_dir):
        self.list_pairs = list_pairs
        self.img_dir = img_dir
        self.transform = trans.Compose([trans.ToTensor(),
                                        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    path = '/workspace/face_rec/dataset/celeba/cropped_img_celeba'
    id_list = os.listdir(path)

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
    
# def get_loader(conf, dataset):
#     ds, class_num = get_dataset(conf.emore_folder/'images'/'train')
#     loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
#     return loader, class_num

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)