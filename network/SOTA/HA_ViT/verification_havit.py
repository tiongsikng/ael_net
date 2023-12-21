import os, sys, copy
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
sys.path.insert(0, os.path.abspath('.'))
from configs import datasets_config as config
import network.SOTA.HA_ViT.HA_ViT as net
from network import load_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 400
eer_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
eer_dict = {}
auc_dict = {}
fpr_dict = {}
tpr_dict = {}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
ver_img_per_class = 4

def compute_eer(fpr,tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = np.around(eer, 4)
    return eer

def create_folder(method):
    lists = ['cm']
    boiler_path = './data/roc/'
    for modal in lists:
        if not os.path.exists(os.path.join(boiler_path, method, modal)):
            os.makedirs(os.path.join(boiler_path, method, modal))

def get_avg(dict_list):
    total_eer = 0
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    for items in dict_list:
        total_eer += dict_list[items]
    dict_list['avg'] = total_eer/len(dict_list)

    return dict_list

class dataset(data.Dataset):
    def __init__(self, dset, root_drt, modal, dset_type='gallery'):
        if modal == 'peri' or modal == 'periocular':
            sz = (112, 112)
        elif modal == 'face':
            sz = (112, 112)
        
        self.ocular_root_dir = os.path.join(os.path.join(root_drt, dset, dset_type), modal[:4])
        self.nof_identity = len(os.listdir(self.ocular_root_dir))
        self.ocular_img_dir_list = []
        self.label_list = []
        self.label_dict = {}
        cnt = 0
        for iden in sorted(os.listdir(self.ocular_root_dir)):
            ver_img_cnt = 0
            for i in range(ver_img_per_class):
                list_img = sorted(os.listdir(os.path.join(self.ocular_root_dir, iden)))
                list_len = len(list_img)
                offset = list_len // ver_img_per_class
                self.ocular_img_dir_list.append(os.path.join(self.ocular_root_dir, iden, list_img[offset*i]))
                self.label_list.append(cnt)
                ver_img_cnt += 1
                if ver_img_cnt == ver_img_per_class:
                    break
            cnt += 1

        self.onehot_label = np.zeros((len(self.ocular_img_dir_list), self.nof_identity))
        for i in range(len(self.ocular_img_dir_list)):
            self.onehot_label[i, self.label_list[i]] = 1

        self.ocular_transform = transforms.Compose([transforms.Resize(sz),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.ocular_img_dir_list)

    def __getitem__(self, idx):
        ocular = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(ocular)
        onehot = self.onehot_label[idx]
        return ocular, onehot

def get_scores(embedding_mat, label_mat):
    score_mat = torch.matmul(embedding_mat, embedding_mat.t()).cpu()
    gen_mat = torch.matmul(label_mat, label_mat.t()).cpu()
    gen_r, gen_c = torch.where(gen_mat == 1)
    imp_r, imp_c = torch.where(gen_mat == 0)

    gen_score = score_mat[gen_r, gen_c].cpu().numpy()
    imp_score = score_mat[imp_r, imp_c].cpu().numpy()

    y_gen = np.ones(gen_score.shape[0])
    y_imp = np.zeros(imp_score.shape[0])

    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((y_gen, y_imp))

    # normalization scores into [ -1, 1]
    score_min = np.amin(score)
    score_max = np.amax(score)
    score = ( score - score_min ) / ( score_max - score_min )
    score = 2.0 * score - 1.0

    return score, y

#### Cross-Modal
def cm_verify(model, emb_size = 512, root_drt=config.evaluation['verification'], device='cuda:0', mode='verify'):
    for dset_name in dset_list:
        embedding_size = emb_size       
        
        if dset_name == 'ethnic':
            peri_dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal='periocular')
            face_dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal='face')
        else:
            peri_dset = dataset(dset=dset_name, dset_type='gallery', root_drt = root_drt, modal='periocular')
            face_dset = dataset(dset=dset_name, dset_type='gallery', root_drt = root_drt, modal='face')

        peri_dloader = torch.utils.data.DataLoader(peri_dset, batch_size=batch_size, num_workers=4)
        nof_peri_dset = len(peri_dset)
        nof_peri_iden = peri_dset.nof_identity
        peri_embedding_mat = torch.zeros((nof_peri_dset, embedding_size)).to(device)
        peri_label_mat = torch.zeros((nof_peri_dset, nof_peri_iden)).to(device)

        face_dloader = torch.utils.data.DataLoader(face_dset, batch_size=batch_size, num_workers=4)
        nof_face_dset = len(face_dset)
        nof_face_iden = face_dset.nof_identity
        face_embedding_mat = torch.zeros((nof_face_dset, embedding_size)).to(device)
        face_label_mat = torch.zeros((nof_face_dset, nof_face_iden)).to(device)

        label_mat = torch.tensor([]).to(device)  


        model = model.eval().to(device)
        with torch.no_grad():
            for i, ((peri_ocular, peri_onehot), (face_ocular, face_onehot)) in enumerate(zip(peri_dloader, face_dloader)):
                nof_peri_img = peri_ocular.shape[0]
                peri_ocular = peri_ocular.to(device)
                peri_onehot = peri_onehot.to(device)
                nof_face_img = face_ocular.shape[0]
                face_ocular = face_ocular.to(device)
                face_onehot = face_onehot.to(device)

                peri_feature, _ = model(peri_ocular.unsqueeze(1), peri_flag=True)
                face_feature, _ = model(face_ocular.unsqueeze(1), peri_flag=False)

                peri_embedding_mat[i*batch_size:i*batch_size+nof_peri_img, :] = peri_feature.detach().clone()                
                peri_label_mat[i*batch_size:i*batch_size+nof_peri_img, :] = peri_onehot
                face_embedding_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_feature.detach().clone()
                face_label_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_onehot         

            label_mat = (face_label_mat)

            ### roc
            face_embedding_mat /= torch.norm(face_embedding_mat, p=2, dim=1, keepdim=True)
            peri_embedding_mat /= torch.norm(peri_embedding_mat, p=2, dim=1, keepdim=True)

            score_mat = torch.matmul(face_embedding_mat, peri_embedding_mat.t()).cpu()
            gen_mat = torch.matmul(label_mat, label_mat.t()).cpu()
            gen_r, gen_c = torch.where(gen_mat == 1)
            imp_r, imp_c = torch.where(gen_mat == 0)

            gen_score = score_mat[gen_r, gen_c].cpu().numpy()
            imp_score = score_mat[imp_r, imp_c].cpu().numpy()

            y_gen = np.ones(gen_score.shape[0])
            y_imp = np.zeros(imp_score.shape[0])

            score = np.concatenate((gen_score, imp_score))
            y = np.concatenate((y_gen, y_imp))

            # normalization scores into [ -1, 1]
            score_min = np.amin(score)
            score_max = np.amax(score)
            score = ( score - score_min ) / ( score_max - score_min )
            score = 2.0 * score - 1.0

            fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
            auc = roc_auc_score(y, score)
            fpr_dict[dset_name] = fpr_tmp
            tpr_dict[dset_name] = tpr_tmp
            auc_dict[dset_name] = auc
            eer_dict[dset_name] = compute_eer(fpr_tmp, tpr_tmp)

    if mode == 'verify':
        return eer_dict
    elif mode == 'roc':
        return eer_dict, fpr_dict, tpr_dict, auc_dict

if __name__ == '__main__':    
    method = 'ha_vit'
    create_folder(method)
    embd_dim = 1024
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    load_model_path = './models/sota/HA-ViT.pth'
    model = net.HA_ViT(img_size=112, patch_size=8, in_chans=3, embed_dim=1024, num_classes_list=(1054,),
                   layer_depth=3, num_heads=8, mlp_ratio=4., norm_layer=None, drop_rate=0., attn_drop_rate=0.,
                   drop_path_rate=0.).to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device) 

    # Verification + ROC Extractor
    cm_eer_dict, cm_fpr_dict, cm_tpr_dict, cm_auc_dict = cm_verify(model, emb_size = embd_dim, root_drt = config.evaluation['verification'], device = device, mode = 'roc')
    cm_eer_dict = get_avg(cm_eer_dict) 
    cm_eer_dict = copy.deepcopy(cm_eer_dict)
    cm_fpr_dict = copy.deepcopy(cm_fpr_dict)
    cm_tpr_dict = copy.deepcopy(cm_tpr_dict)
    cm_auc_dict = copy.deepcopy(cm_auc_dict)    
    torch.save(cm_eer_dict, './data/roc/' + str(method) + '/cm/cm_eer_dict.pt')
    torch.save(cm_fpr_dict, './data/roc/' + str(method) + '/cm/cm_fpr_dict.pt')
    torch.save(cm_tpr_dict, './data/roc/' + str(method) + '/cm/cm_tpr_dict.pt')
    torch.save(cm_auc_dict, './data/roc/' + str(method) + '/cm/cm_auc_dict.pt') 
    print('Cross-modal:', cm_eer_dict)
    print('Average:', cm_eer_dict['avg'])