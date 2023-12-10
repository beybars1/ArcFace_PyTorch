import torch, os, time, sys
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from models import Backbone, Arcface


device = int(os.environ["LOCAL_RANK"])
init_process_group(backend="nccl")

model = Backbone(50, 0.6, 'ir_se').to(device)
head = Arcface(embedding_size=512, classnum=400000).to(device)

model = DDP(model, device_ids=[device])
head = DDP(head, device_ids=[device])

map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
snapshot = torch.load('/data/dev_5/beybars/face_rec/personal_arcFace/distributed_appr/workspace/steps_models_save/model_2023-01-20-16-48_accuracy:0.98695_epoch:0_step:183960_None.pth', map_location=map_location)         
# if device == 0:
#     print(snapshot.keys())
#     print(snapshot)
from collections import OrderedDict
new_state_dict_model = OrderedDict()
new_state_dict_head = OrderedDict()

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

model.load_state_dict(new_state_dict_model)
head.load_state_dict(new_state_dict_head)





destroy_process_group()















# import torch, os, time, sys
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# from models import Backbone, Arcface
# from collections import OrderedDict


# device = int(os.environ["LOCAL_RANK"])
# init_process_group(backend="nccl")

# model = Backbone(50, 0.6, 'ir_se').to(device)
# head = Arcface(embedding_size=512, classnum=400000).to(device)

# model = DDP(model, device_ids=[device])
# head = DDP(head, device_ids=[device])

# map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
# snapshot = torch.load('/data/dev_5/beybars/face_rec/personal_arcFace/distributed_appr/workspace/steps_models_save/model_2023-01-20-16-48_accuracy:0.98695_epoch:0_step:183960_None.pth', map_location=map_location)         
# model.module.load_state_dict(snapshot['MODEL_STATE'])
# head.module.load_state_dict(snapshot["HEAD_STATE"])

# # if device == 0:
# #     print(snapshot.keys())
# #     print(snapshot)
# # new_state_dict_model = OrderedDict()
# # new_state_dict_head = OrderedDict()

# # for k, v in snapshot['MODEL_STATE'].items():
# #     if 'module' not in k:
# #         k = 'module.'+k
# #     else:
# #         k = k.replace('features.module.', 'module.features.')
# #     new_state_dict_model[k]=v
    
# # for k, v in snapshot["HEAD_STATE"].items():
# #     if 'module' not in k:
# #         k = 'module.'+k
# #     else:
# #         k = k.replace('features.module.', 'module.features.')
# #     new_state_dict_head[k]=v

# # model.load_state_dict(new_state_dict_model)
# # head.load_state_dict(new_state_dict_head)

# print('wait for 20 secs')
# time.sleep(20)
# destroy_process_group()

















