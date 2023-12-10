from easydict import EasyDict as edict
from pathlib import Path
import torch, os
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.resume_training_steps = True
    conf.resume_training_epochs = False
    #model_2023-01-25-09-30_accuracy:0.9863500000000001_epoch:0_step:165564_None
    #conf.checkpoint_name = '2023-02-08-03-39_accuracy:0.9972624999999999_epoch:6_step:153300_None.pth'
    conf.checkpoint_name = '2023-02-18-14-44_accuracy:0.9975999999999999_epoch:11_step:86235_None.pth'
    conf.data_path = '/data/beybars_id/'
    conf.work_path = '/data/dev_5/beybars/face_rec/personal_arcFace/distributed_appr/workspace/'
    conf.steps_model_path = conf.work_path+'steps_models_save/'
    conf.log_path = conf.work_path+'log13/'
    conf.epochs_model_path = conf.work_path+'epochs_models_save/'
    conf.input_size = [112, 112] # cropped image shape
    conf.embedding_size = 512 # output embedding shape
    conf.use_mobilfacenet = False
    conf.net_depth = 100 # 50, 100, 152 depth
    conf.drop_ratio = 0.6 # drop out probablity at FC layer
    conf.net_mode = 'ir_se' # or 'ir', either use Imporved ResNet + Squeeze and Excitation or Improved ResNet
    #conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.world_size = torch.cuda.device_count()
    conf.device = int(os.environ["LOCAL_RANK"])
    conf.board_loss_num = 3000
    conf.evaluate_num = 80
    conf.save_num = 40
    
    
    # TODO: need to calculate own means/stdvs
    conf.test_transform = trans.Compose([trans.ToTensor(), trans.Normalize([0.4864, 0.5090, 0.4477], [0.2088, 0.1916, 0.1992])])
    conf.data_mode = 'celeba'
    conf.celeba_folder = conf.data_path+'img_data/'
    conf.celeba_train = conf.celeba_folder+'cropped_alligned/'
    conf.celeba_val = conf.celeba_folder+'cropped_alligned_val/'

    conf.k_pairs = 12
    conf.sim_metric = 'euclidean_dis'
    conf.batch_size = 128 # irse net depth 50
    #conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        #conf.log_path = conf.work_path+'log5/'
        conf.steps_model_path = conf.work_path+'steps_models_save/'
        #conf.weight_decay = 5e-4
        conf.lr = 0.1 # original 1e-3
        conf.milestones = [12, 14, 16] # original [12, 15, 18]
        conf.momentum = 0.9
        conf.pin_memory = True
        #conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 4
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        #the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf



# tensor([0.4864, 0.5090, 0.4477])
# tensor([0.2088, 0.1916, 0.1992])