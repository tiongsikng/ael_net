import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import os, sys, copy
sys.path.insert(0, os.path.abspath('.'))
from data.data_loader import ConvertRGB2BGR, FixedImageStandard
from network import load_model
from configs import datasets_config as config
import network.ael_net as net

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 400
eer_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
auc_dict = {}
fpr_dict = {}
tpr_dict = {}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
ver_img_per_class = 4


def create_folder(method):
    lists = ['peri', 'face', 'cm']
    boiler_path = './data/roc/'
    for modal in lists:
        if not os.path.exists(os.path.join(boiler_path, method, modal)):
            os.makedirs(os.path.join(boiler_path, method, modal))


def get_avg(dict_list):
    total_eer = 0
    eer_list = []
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    if 'std' in dict_list.keys():
        del dict_list['std']
    for items in dict_list:
        total_eer += dict_list[items]
        eer_list.append(dict_list[items])
    dict_list['avg'] = total_eer/len(dict_list) * 100
    dict_list['std'] = np.std(np.array(eer_list)) * 100

    return dict_list


def eer_calc(gen_dist, imp_dist):
    gen_dist, imp_dist = np.array(gen_dist), np.array(imp_dist)
    gens = np.ones(len(gen_dist))
    imps = np.zeros(len(imp_dist))

    lbl_lst = np.hstack((gens, imps))
    dist_lst = np.hstack((gen_dist, imp_dist))

    far_, tar_, threshold = roc_curve(lbl_lst, dist_lst, pos_label=1)
    frr_ = 1 - tar_
    eer_ = far_[np.nanargmin(np.absolute((frr_ - far_)))]
    auc_ = roc_auc_score(lbl_lst, dist_lst)

    return eer_, far_, tar_, auc_


def compute_eer(fpr,tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = np.around(eer, 4)
    return eer


class dataset(data.Dataset):
    def __init__(self, dset, root_drt, modal, dset_type='gallery'):
        if modal == 'peri' or modal == 'periocular':
            sz = (112, 112) # (37, 112)
        elif modal == 'face':
            sz = (112, 112)
        
        self.ocular_root_dir = os.path.join(os.path.join(root_drt, dset, dset_type), modal)
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
                                     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    def __len__(self):
        return len(self.ocular_img_dir_list)

    def __getitem__(self, idx):
        ocular = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(ocular)
        onehot = self.onehot_label[idx]
        return ocular, onehot


#### Cross-Modal ROC
def cm_verify(model, face_model, peri_model, emb_size=1024, root_drt=config.evaluation['verification'], device='cuda:0', eval_mode='verify'):
    for dset_name in dset_list:
        embedding_size = emb_size       
        
        if dset_name == 'ethnic':
            peri_dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal='peri')
            face_dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal='face')
        else:
            peri_dset = dataset(dset=dset_name, dset_type='gallery', root_drt = root_drt, modal='peri')
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
            for i, (peri_ocular, peri_onehot) in enumerate(peri_dloader):
                nof_peri_img = peri_ocular.shape[0]
                peri_ocular = peri_ocular.to(device)
                peri_onehot = peri_onehot.to(device)

                if not peri_model is None:
                    peri_feature = peri_model(peri_ocular, peri_flag = True)
                else:
                    peri_feature = model(peri_ocular, peri_flag = True)

                peri_embedding_mat[i*batch_size:i*batch_size+nof_peri_img, :] = peri_feature.detach().clone()                
                peri_label_mat[i*batch_size:i*batch_size+nof_peri_img, :] = peri_onehot


            for i, (face_ocular, face_onehot) in enumerate(face_dloader):
                nof_face_img = face_ocular.shape[0]
                face_ocular = face_ocular.to(device)
                face_onehot = face_onehot.to(device)

                if not face_model is None:
                    face_feature = face_model(face_ocular, peri_flag = False)
                else:
                    face_feature = model(face_ocular, peri_flag = False)

                face_embedding_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_feature.detach().clone()
                face_label_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_onehot           

            embedding_mat = torch.cat((face_embedding_mat, peri_embedding_mat), 1)  
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

            fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
            auc = roc_auc_score(y, score)
            fpr_dict[dset_name] = fpr_tmp
            tpr_dict[dset_name] = tpr_tmp
            auc_dict[dset_name] = auc
            eer_dict[dset_name] = compute_eer(fpr_tmp, tpr_tmp)

    if eval_mode == 'roc':
        return eer_dict, fpr_dict, tpr_dict, auc_dict
    elif eval_mode == 'verify':
        return eer_dict


if __name__ == '__main__':
    method = 'AELNet'
    eval_mode = 'roc' # ROC (graph) or verification (values)
    if eval_mode == 'roc':
        create_folder(method)
    embd_dim = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_model_path = './models/best_model/AELNet.pth'
    model = net.AEL_Net(embedding_size = embd_dim, do_prob=0.0).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)

    cm_eer_dict, cm_fpr_dict, cm_tpr_dict, cm_auc_dict = cm_verify(model, face_model=None, peri_model=None, emb_size=embd_dim, root_drt=config.evaluation['verification'], device=device, eval_mode=eval_mode)
    if eval_mode == 'roc':
        cm_eer_dict = copy.deepcopy(cm_eer_dict)
        cm_fpr_dict = copy.deepcopy(cm_fpr_dict)
        cm_tpr_dict = copy.deepcopy(cm_tpr_dict)
        cm_auc_dict = copy.deepcopy(cm_auc_dict)
        torch.save(cm_eer_dict, './data/roc/' + str(method) + '/cm/cm_eer_dict.pt')
        torch.save(cm_fpr_dict, './data/roc/' + str(method) + '/cm/cm_fpr_dict.pt')
        torch.save(cm_tpr_dict, './data/roc/' + str(method) + '/cm/cm_tpr_dict.pt')
        torch.save(cm_auc_dict, './data/roc/' + str(method) + '/cm/cm_auc_dict.pt')
    cm_eer_dict = get_avg(cm_eer_dict)
    print('Average EER (Cross-Modal):', cm_eer_dict['avg'], '±', cm_eer_dict['std'])