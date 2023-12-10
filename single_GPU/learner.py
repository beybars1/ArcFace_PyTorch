from dataloader import create_pairs, create_train_list, DatasetTrain, DatasetValidation
from torch.utils.data import DataLoader
from models import Backbone, Arcface, MobileFaceNet
import torch.nn.functional as F
from verification import evaluate_metrics
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz

class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)

        # Backbone selection
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            # Backbone Model
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        # TRAINING MODE
        if not inference:
            
            # PATHLIB bug-issue
            #self.train_dir = list(map(str, conf.celeba_train))
            #self.val_dir = list(map(str, conf.celeba_val))

            # train dataset
            self.train_list, labels = create_train_list(conf.celeba_train)
            #print(labels)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.train_dataset = DatasetTrain(self.train_list, conf.celeba_train, self.labels)
            self.train_loader = DataLoader(self.train_dataset, batch_size=conf.batch_size,
                                                                shuffle=True,
                                                                pin_memory=conf.pin_memory,
                                                                num_workers=conf.num_workers)
            self.class_num = self.train_dataset.__len__()
            # TensorBoardX
            self.writer = SummaryWriter(conf.log_path)

            #ArcFace Model Head
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            print('Backbone and ArcFace head are generated ...')
            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            # important initializations
            self.milestones = conf.milestones
            self.step = 0
            self.board_loss_every = len(self.train_loader)//100
            self.evaluate_every = len(self.train_loader)//10
            self.save_every = len(self.train_loader)//5

            # Optimizer
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)
            #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)
            print('Optimizer generated...') 
            
            # validation dataset
            self.id_list = os.listdir(conf.celeba_val)
            self.list_pairs = create_pairs(self.id_list, conf.celeba_val, conf.k_pairs)
            self.val_dataset = DatasetValidation(self.list_pairs, conf.celeba_val)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=conf.batch_size,
                                                                shuffle=True,
                                                                pin_memory=conf.pin_memory,
                                                                num_workers=conf.num_workers)
        else:
            self.threshold = conf.threshold

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        # self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
        # self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
        # self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(self.model.state_dict(),\
            save_path+'/'+('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(),\
                    save_path+'/'+('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(),\
                    save_path+'/'+('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        self.model.load_state_dict(torch.load(save_path+'/'+'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path+'/'+'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path+'/'+'optimizer_{}'.format(fixed_str)))

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        num_correct = 0      
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()
                                          
            for train_batch in tqdm(self.train_loader):
                imgs = train_batch['IMG_1']
                labels = train_batch['LABEL']
                #imgs_name = train_batch['IMG_1_NAME']
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                max_val, _ = torch.max(thetas, dim=1)
                num_correct += (max_val == labels).float().sum()
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss (CE)', loss_board, self.step)
                    running_loss = 0.
                    acc_board = num_correct / self.board_loss_every
                    self.writer.add_scalar('train_accuracy', acc_board, self.step)
                    num_correct = 0
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, nrof_folds=10)
                    self.board_val('celeba', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                self.step += 1
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def evaluate(self, conf, nrof_folds=10):
        self.model.eval()
        print("Validation...")
        emb_list_1 = []
        emb_list_2 = []
        isSame_list = []#################################################
        with torch.no_grad():
            for val_batch in self.val_dataloader:
                imgs_1 = val_batch['IMG_1']
                imgs_2 = val_batch['IMG_2']
                isSame = val_batch['isSame']
                imgs_1 = imgs_1.to(conf.device)
                imgs_2 = imgs_2.to(conf.device)
                embeddings_1 = self.model(imgs_1)
                embeddings_2 = self.model(imgs_2)
                embeddings_normed_1 = F.normalize(embeddings_1)
                embeddings_normed_2 = F.normalize(embeddings_2)
                embeddings_normed_1 = embeddings_normed_1.detach().cpu()
                embeddings_normed_2 = embeddings_normed_2.detach().cpu()
                emb_list_1.append(embeddings_normed_1)
                emb_list_2.append(embeddings_normed_2)
                isSame_list+=isSame#################################################
            embeddings_tensor_1 = torch.vstack(emb_list_1)
            embeddings_tensor_2 = torch.vstack(emb_list_2)
            TPR, TNR, FPR, FNR, accuracy, best_thresholds = evaluate_metrics(embeddings_tensor_1, embeddings_tensor_2,\
                                                                            isSame_list, conf.sim_metric, nrof_folds)#################################################
            #print(TP.shape)
            # TPR = 0 if (TP + FN == 0) else float(TP) / float(TP + FN)
            # TNR = 0 if (TN + FP == 0) else float(TN) / float(TN + FP)
            # FPR = 0 if (FP + TN == 0) else float(FP) / float(FP + TN)
            # FNR = 0 if (FN + TP == 0) else float(FN) / float(FN + TP)
            buffer = gen_plot(FPR, TPR)
            roc_curve = Image.open(buffer)
            roc_curve_tensor = trans.ToTensor()(roc_curve)
            return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(F.normalize(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum

    

