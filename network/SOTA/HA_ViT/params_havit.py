import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
seed = 42
save = True
write_log = True
send_msg = False
send_log = False
method = 'HA-ViT'
remarks = 'Reproduce_Search_LargeMargin'
dim = 1024

# Other hyperparameters
single_param_type = 2 # 2 for NOMS, 3 for NMS, 4 for OMS
batch_sub = 16
batch_samp = 1
face_batch_size = batch_sub * batch_samp
random_batch_size = 16
test_batch_size = 80 #face_batch_size
epochs = 40
epochs_pre = 0
ecc_iters = 25

lr = 1e-4 #0.001
lr_sch = [12, 24, 36] #[6, 18, 30, 42]
w_decay = 1e-5
dropout = 0.1
momentum = 0.9

# Ensemble
weight_1 = 1.0
weight_2 = 1.0
weight_3 = 1.0
weight_4 = 1.0

# LDPC ECC
N = 52
m = 42
code_n = N
code_k = N - m
code_rate = 1.0 * (N - m) / (N-2)
Z = 16 # 16, 3, or 10
numOfWordSim_train = face_batch_size
batch_size = numOfWordSim_train
num_of_batch = 5000

# Index-of-Max Hashing
iom_cfg = {'m': 1024, 'q': 8, 'T': 1}

# Activate, or deactivate BatchNorm2D
# bn_flag = 0, 1, 2
bn_flag = 1
bn_moment = 0.1
if bn_flag == 1:
    bn_moment = 0.1

# Softmax Classifiers
af_s = 64
af_m = 0.35
cf_s = 64
cf_m = 0.35

# Triplet Loss (set tl_id to -1 to use AP)
tl_id = 1
tl_m = 1.0
tl_k = 1
tl_alpha = 10
tl_ap = 0.001 # 0.01++
if tl_id >= 0:
    tl_ap = 0.0

# Activate / deactivate face_fc, peri_fc w.r.t. face_fc_ce_flag, peri_fc_ce_flag + network description
face_fc_ce_flag = True
peri_fc_ce_flag = True
face_peri_loss_flag = True

if face_fc_ce_flag is True and peri_fc_ce_flag is True and face_peri_loss_flag is False:
    net_descr = 'face_fc w/ Softmax + peri w/ Softmax'
    net_tag = str('11_0')
    tl_id = 0
    tl_m = 0.0
    tl_k = 0.0
    tl_alpha = 0.0
    b2_flag = False
elif face_fc_ce_flag is True and peri_fc_ce_flag is True and face_peri_loss_flag is True:
    net_descr = 'face_fc w/ Softmax + peri w/ Softmax + Modality Alignment Loss'
    net_tag = str('11_1')
    b2_flag = False
elif face_fc_ce_flag is False and peri_fc_ce_flag is True and face_peri_loss_flag is False:
    net_descr = 'Baseline: peri_fc w/ Softmax'
    net_tag = str('01_0B')
    tl_id = 0
    tl_m = 0.0
    tl_k = 0.0
    tl_alpha = 0.0
elif face_fc_ce_flag is True and peri_fc_ce_flag is False and face_peri_loss_flag is False:
    net_descr = 'Baseline: face_fc w/ Softmax'
    net_tag = str('10_0B')
    tl_id = 0
    tl_m = 0.0
    tl_k = 0.0
    tl_alpha = 0.0
else:
    net_descr = 'unknown'
    raise ValueError("Unknown network parameters.")


bn_moment = float(bn_moment)
dropout = float(dropout)
af_s = float(af_s)
af_m = float(af_m)
cf_s = float(cf_s)
cf_m = float(cf_m)
tl_id = int(tl_id)
tl_m = float(tl_m)
tl_k = int(tl_k)
tl_alpha = float(tl_alpha)