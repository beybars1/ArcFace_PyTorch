from dataloader import create_pairs, create_train_list, DatasetTrain, DatasetValidation
from torch.utils.data import DataLoader
from models import Backbone, Arcface, MobileFaceNet
import torch.nn.functional as F
from verification import evaluate_metrics
import torch, sys
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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
torch.manual_seed(44)



class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        
        self.ddp_setup() # start DDP session
        
        # Backbone selection
        # if conf.use_mobilfacenet:
        #     self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
        #     print('MobileFaceNet model generated')
        # else:
        
        if not inference:
            # train dataset
            self.train_list, labels, self.class_num = create_train_list(conf.celeba_train)
            #print(self.class_num)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.train_dataset = DatasetTrain(self.train_list, conf.celeba_train, self.labels)
            self.train_loader = DataLoader(self.train_dataset, batch_size=conf.batch_size,
                                           shuffle=False, pin_memory=conf.pin_memory,
                                           num_workers=conf.num_workers, sampler=DistributedSampler(self.train_dataset, shuffle=True, seed=1))
                   
            # TensorBoardX
            self.writer = SummaryWriter(conf.log_path)
            
            self.milestones = conf.milestones
            self.step = 0
            self.epoch = 0
            self.board_loss_every = len(self.train_loader)//conf.board_loss_num
            self.evaluate_every = len(self.train_loader)//conf.evaluate_num
            self.save_every = len(self.train_loader)//conf.save_num
        
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            self.model = DDP(self.model, device_ids=[conf.device])
            self.head = DDP(self.head, device_ids=[conf.device])
            
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
            print('Backbone and ArcFace head are generated ...')
            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            # Optimizer
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.module.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
                #self.optimizer = DDP(self.optimizer, device_ids=conf.device)
            print(self.optimizer)
            #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)
            print('Optimizer generated...')
            
            
            if conf.resume_training_steps:
                self.load_state(conf.steps_model_path, conf.checkpoint_name, conf.device, steps_save=True)
                print("Resuming training from epoch {}".format(self.epoch))
            elif conf.resume_training_epochs:
                self.load_state(conf.epochs_model_path, conf.checkpoint_name, steps_save=False)
                print("Resuming training from epoch {}".format(self.epoch))
            else:
                pass # from scratch
            
            
            
            
            # validation dataset
            self.id_list = os.listdir(conf.celeba_val)
            self.list_pairs = create_pairs(self.id_list, conf.celeba_val, conf.k_pairs)
            self.val_dataset = DatasetValidation(self.list_pairs, conf.celeba_val)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=conf.batch_size,
                                             shuffle=False, pin_memory=conf.pin_memory,
                                             num_workers=conf.num_workers, sampler=DistributedSampler(self.val_dataset, shuffle=True, seed=1))
            
            
            
        else:
            self.threshold = conf.threshold

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
        
    def ddp_setup(self):
        init_process_group(backend="nccl")

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_val_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        # self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
        # self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
        # self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def save_state(self, conf, accuracy, steps_save=True, extra=None):
        if steps_save:
            save_path = conf.steps_model_path
            snapshot = {}
            snapshot["STEPS_RUN"] = self.step
            snapshot["EPOCHS_RUN"] = self.epoch
            snapshot["MODEL_STATE"] = self.model.module.state_dict()
            snapshot["HEAD_STATE"] = self.head.module.state_dict()
            snapshot["OPTIMIZER"] = self.optimizer.state_dict()
            torch.save(snapshot, save_path+'/'+('model_{}_accuracy:{}_epoch:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.epoch, self.step, extra)))
            
        else:
            save_path = conf.epochs_model_path
            snapshot = {}
            snapshot["EPOCHS_RUN"] = self.epoch
            snapshot["MODEL_STATE"] = self.model.module.state_dict()
            snapshot["HEAD_STATE"] = self.head.module.state_dict()
            snapshot["OPTIMIZER"] = self.optimizer.state_dict()
            torch.save(snapshot, save_path+'/'+('model_{}_accuracy:{}_epoch:{}_{}.pth'.format(get_time(), accuracy, self.epoch, extra)))
        
        
    def load_state(self, model_path, checkpoint_name, rank, steps_save=True):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        
        from collections import OrderedDict
        new_state_dict_model = OrderedDict()
        new_state_dict_head = OrderedDict()
        
        if steps_save:
            #print("here !!!!!!!!!!!!!!!!")
            #print(type(save_path))
            snapshot = torch.load(model_path+'/'+'model_{}'.format(checkpoint_name), map_location=map_location)
            for k, v in snapshot['MODEL_STATE'].items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict_model[k]=v
            
            for k, v in snapshot["HEAD_STATE"].items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict_head[k]=v
            
            self.model.load_state_dict(new_state_dict_model)
            self.head.load_state_dict(new_state_dict_head)      
            # self.model.module.load_state_dict(snapshot['MODEL_STATE'])
            # self.head.module.load_state_dict(snapshot["HEAD_STATE"])
            self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
            self.epoch = snapshot["EPOCHS_RUN"]+1
        else:
            save_path = model_path
            snapshot = torch.load(save_path+'/'+'model_{}'.format(checkpoint_name))         
            self.model.load_state_dict(snapshot['MODEL_STATE'])
            self.head.load_state_dict(snapshot["HEAD_STATE"])
            self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
            self.epoch = snapshot["EPOCHS_RUN"]+1

    def train(self, conf, epochs):
        self.model.train()
        running_loss = torch.zeros((1))
        # num_correct = torch.zeros((1))
        # total_samples = 0
        for e in range(self.epoch, epochs):
            self.epoch = e
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()
            # tmp = torch.ones((5,5)).to(conf.device)##########################################################################
            for train_batch in tqdm(self.train_loader):
                running_loss = running_loss.to(conf.device)
                imgs = train_batch['IMG_1']
                labels = train_batch['LABEL']
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                
                # max_val, max_idx = torch.max(thetas, dim=1)
                # max_idx = max_idx.to(conf.device)
                # total_samples += thetas.shape[0]
                # num_correct += (max_idx == labels).sum()
               

                loss = conf.ce_loss(thetas, labels)
                #print("loss", loss)
                loss.backward()
                running_loss += loss
                #print("run loss", running_loss)
                # torch.distributed.all_reduce(tmp, torch.distributed.ReduceOp.SUM)######################################################################
                # tmp_avg = tmp/conf.world_size###########################################################################################################################
                # print(tmp_avg)#############################################################################################################
                #print("Device", conf.device, "loss:", loss, '\n')
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    torch.distributed.all_reduce(running_loss, torch.distributed.ReduceOp.SUM)
                    avg_loss_from_proc = running_loss/conf.world_size
                    # torch.distributed.all_reduce(num_correct, torch.distributed.ReduceOp.SUM)
                    # average_acc_from_proc = num_correct/conf.world_size
                    
                    if conf.device==0:
                        #print('board', self.step)
                        loss_board = avg_loss_from_proc / self.board_loss_every
                        print("lossBoard", loss_board)
                        self.writer.add_scalar('train_loss (CE)', loss_board, self.step)
                        running_loss = torch.zeros((1))
                        # acc_board = average_acc_from_proc / total_samples
                        # self.writer.add_scalar('train_accuracy (CLS-softmax)', acc_board, self.step)
                        # num_correct = torch.zeros((1))
                    if conf.device in [1,2,3]:
                        running_loss = torch.zeros((1))
                        
                if self.step % self.evaluate_every == 0 and self.step != 0 and conf.device == 0:
                    #print("eval", self.step)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, nrof_folds=5)
                    self.board_val('celeba', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                    
                if self.step % self.save_every == 0 and self.step != 0 and conf.device == 0:
                    #print('save', self.step)
                    self.save_state(conf, accuracy)
                    
                self.step += 1
            if conf.device==0:
                self.save_state(conf, accuracy, steps_save=False, extra=None)
        if conf.device==0:
            self.save_state(conf, accuracy, steps_save=False, extra='final')
            
        # end DDP session
        destroy_process_group()
        
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
                embeddings_1 = self.model.module(imgs_1)
                embeddings_2 = self.model.module(imgs_2)
                embeddings_normed_1 = F.normalize(embeddings_1)
                embeddings_normed_2 = F.normalize(embeddings_2)
                embeddings_normed_1 = embeddings_normed_1.detach().cpu()
                embeddings_normed_2 = embeddings_normed_2.detach().cpu()
                emb_list_1.append(embeddings_normed_1)
                emb_list_2.append(embeddings_normed_2)
                isSame_list+=isSame#################################################
            embeddings_tensor_1 = torch.vstack(emb_list_1)
            embeddings_tensor_2 = torch.vstack(emb_list_2)
            TPR, TNR, FPR, FNR, accuracy, best_thresholds, tp, tn, fp, fn = evaluate_metrics(embeddings_tensor_1, embeddings_tensor_2,\
                                                                                             isSame_list, conf.sim_metric, nrof_folds)
            if conf.device==0:
              thresholds = np.arange(0, 4, 0.01)
              for thr in best_thresholds:
                indx=np.where(thresholds == thr)
                ind = indx[0].item()
                precision = tp[ind]/(tp[ind]+fp[ind])
                recall = tp[ind]/(tp[ind]+fn[ind])
                fraud = tn[ind]/(tn[ind]+fp[ind])
                print('threshold:{},\n precision:{},\n recall:{},\n fraud_det:{}'.format(thr, precision, recall, fraud))
                print()
            #print(TP.shape)
            # TPR = 0 if (TP + FN == 0) else float(TP) / float(TP + FN)
            # TNR = 0 if (TN + FP == 0) else float(TN) / float(TN + FP)
            # FPR = 0 if (FP + TN == 0) else float(FP) / float(FP + TN)
            # FNR = 0 if (FN + TP == 0) else float(FN) / float(FN + TP)
            
            from sklearn import metrics
            auc_roc = metrics.auc(FPR,TPR)
            print("auc_roc:", auc_roc)
            # TPR=TPR.to(conf.device)
            # TNR=TNR.to(conf.device)
            # FPR=FPR.to(conf.device)
            # FNR=FNR.to(conf.device)
            # accuracy=accuracy.to(conf.device)
            # best_thresholds=best_thresholds.to(conf.device)
            
            # torch.distributed.all_reduce(TPR, torch.distributed.ReduceOp.SUM)
            # average_TPR = TPR/conf.world_size
            # torch.distributed.all_reduce(TNR, torch.distributed.ReduceOp.SUM)
            # average_TNR = TNR/conf.world_size
            # torch.distributed.all_reduce(FPR, torch.distributed.ReduceOp.SUM)
            # average_FPR = FPR/conf.world_size
            # torch.distributed.all_reduce(FNR, torch.distributed.ReduceOp.SUM)
            # average_FNR = FNR/conf.world_size
            
            # torch.distributed.all_reduce(accuracy, torch.distributed.ReduceOp.SUM)
            # average_accuracy = accuracy/conf.world_size
            # torch.distributed.all_reduce(best_thresholds, torch.distributed.ReduceOp.SUM)
            # average_best_thresholds = best_thresholds/conf.world_size
            # average_FPR=average_FPR.detach().cpu()
            # average_TPR=average_TPR.detach().cpu()
            #buffer = gen_plot(FPR, TPR)
            buffer = gen_plot(FPR, TPR)
            roc_curve = Image.open(buffer)
            roc_curve_tensor = trans.ToTensor()(roc_curve)
            #return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
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

    

