import copy
from sklearn.metrics import pairwise
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import os, glob, sys
sys.path.insert(0, os.path.abspath('.'))
import network.ael_net as net
from network import load_model
from configs import datasets_config as config
from data import data_loader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

cmc_dict = {}
cmc_avg_dict = {}
cm_cmc_dict_p = {}
cm_cmc_avg_dict_p = {}
cm_cmc_dict_f = {}
cm_cmc_avg_dict_f = {}
id_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_id_dict_f = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_id_dict_p = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
dset_name = ['Ethnic', 'Pubfig', 'FaceScrub', 'IMDb Wiki', 'AR']


def create_folder(method):
    lists = ['peri', 'face', 'cm']
    boiler_path = './data/cmc/'
    for modal in lists:
        if not os.path.exists(os.path.join(boiler_path, method, modal)):
            os.makedirs(os.path.join(boiler_path, method, modal))


def get_avg(dict_list):
    total_ir = 0
    ir_list = []
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    if 'std' in dict_list.keys():
        del dict_list['std']
    for items in dict_list:
        total_ir += dict_list[items]
        ir_list.append(dict_list[items])
    dict_list['avg'] = total_ir/len(dict_list)
    dict_list['std'] = np.std(np.array(ir_list)) * 100

    return dict_list


#### Cross-Modal Identification (Main)
def cm_id_main(model, root_pth=config.evaluation['identification'], face_model=None, peri_model=None, device='cuda:0'):
    for datasets in dset_list:

        root_drt = root_pth + datasets + '/**'
        modal_root = ['/peri/', '/face/']
        path_lst = []
        data_loaders = []
        acc_face_gal = []
        acc_peri_gal = []    

        # *** ***

        if datasets == 'ethnic':
            ethnic_face_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', 'face', aug='False')
            ethnic_peri_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', 'periocular', aug='False')
            acc_face_gal = crossmodal_id(model, ethnic_face_gal_load, ethnic_peri_pr_load, device=device, face_model=face_model, peri_model=peri_model, gallery='face')

            ethnic_peri_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', 'periocular', aug='False')
            ethnic_face_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', 'face', aug='False')
            acc_peri_gal = crossmodal_id(model, ethnic_face_pr_load, ethnic_peri_gal_load, device=device, face_model=face_model, peri_model=peri_model, gallery='peri')
        else:
            # data loader and datasets
            for directs in glob.glob(root_drt):
                base_nm = directs.split('\\')[-1]
                if not directs.split('/')[-1] == 'gallery':
                    path_lst.append(directs)
                else:
                    gallery_path = directs      

            fold = 0
            for probes in path_lst:
                fold += 1
                peri_probe_load, peri_dataset = data_loader.gen_data((probes + modal_root[0]), 'test', 'periocular', aug='False')
                face_gal_load, face_dataset = data_loader.gen_data((gallery_path + modal_root[1]), 'test', 'face', aug='False')
                cm_face_gal_acc = crossmodal_id(model, face_gal_load, peri_probe_load, device=device, face_model=face_model, peri_model = peri_model, gallery='face')
                cm_face_gal_acc = np.around(cm_face_gal_acc, 4)
                acc_face_gal.append(cm_face_gal_acc)

                peri_gal_load, peri_dataset = data_loader.gen_data((gallery_path + modal_root[0]), 'test', 'periocular', aug='False')
                face_probe_load, face_dataset = data_loader.gen_data((probes + modal_root[1]), 'test', 'face', aug='False')
                cm_peri_gal_acc = crossmodal_id(model, face_probe_load, peri_gal_load, device=device, face_model=face_model, peri_model=peri_model, gallery='peri')
                cm_peri_gal_acc = np.around(cm_peri_gal_acc, 4)
                acc_peri_gal.append(cm_peri_gal_acc)

        # *** ***

        acc_peri_gal = np.around(np.mean(acc_peri_gal), 4)
        acc_face_gal = np.around(np.mean(acc_face_gal), 4)        
        print('Peri Gallery:', datasets, acc_peri_gal)
        print('Face Gallery:', datasets, acc_face_gal) 
        cm_id_dict_p[datasets] = acc_peri_gal       
        cm_id_dict_f[datasets] = acc_face_gal        

    return cm_id_dict_p, cm_id_dict_f


#### Cross-Modal Identification. For baselines, input face_model and peri_model
def crossmodal_id(model, face_loader, peri_loader, device='cuda:0', face_model=None, peri_model=None, gallery='face'):
    
    # ***** *****    
    model = model.eval().to(device)
    # ***** *****

    # Extract face features w.r.t. pre-learned model
    face_fea = torch.tensor([])
    face_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag=False)
            else:
                x = model(x, peri_flag=False)

            face_fea = torch.cat((face_fea, x.detach().cpu()), 0)
            face_label = torch.cat((face_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    # print('Test Set Capacity\t: ', test_fea.size())
    assert(face_fea.size()[0] == face_label.size()[0])
    
    del face_loader
    time.sleep(0.0001)

    # *****    
    
    # Extract periocular features w.r.t. pre-learned model
    peri_fea = torch.tensor([])
    peri_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(peri_loader):

            x = x.to(device)
            if not peri_model is None:
                peri_model = peri_model.eval().to(device)
                x = peri_model(x, peri_flag=True)
            else:
                x = model(x, peri_flag=True)

            peri_fea = torch.cat((peri_fea, x.detach().cpu()), 0)
            peri_label = torch.cat((peri_label, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(peri_fea.size()[0] == peri_label.size()[0])
    
    del peri_loader
    time.sleep(0.0001)
    
    # ***** *****
    # perform checking
    if gallery == 'face':
        gal_fea, gal_label = face_fea, face_label
        probe_fea, probe_label = peri_fea, peri_label
    elif gallery == 'peri':
        gal_fea, gal_label = peri_fea, peri_label
        probe_fea, probe_label = face_fea, face_label

    # normalize features
    gal_fea = F.normalize(gal_fea, p=2, dim=1)
    probe_fea = F.normalize(probe_fea, p=2, dim=1)

    # Calculate gallery_acc and test_acc
    gal_label = np.reshape(np.array(gal_label), -1)
    probe_label = np.reshape(np.array(probe_label), -1)    
    
    probe_dist = pairwise.cosine_similarity(gal_fea, probe_fea)
    probe_pred = np.argmax(probe_dist, 0)
    probe_pred = gal_label[probe_pred]
    probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]
    
    del model
    time.sleep(0.0001)
    
    return probe_acc


def feature_extractor(model, data_loader, device='cuda:0', peri_flag=False):    
    emb = torch.tensor([])
    lbl = torch.tensor([], dtype = torch.int64)

    model = model.eval().to(device)
    
    with torch.no_grad():        
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

            emb = torch.cat((emb, x.detach().cpu()), 0)
            lbl = torch.cat((lbl, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Set Capacity\t: ', emb.size())
    assert(emb.size()[0] == lbl.size()[0])
    
    del data_loader
    time.sleep(0.0001)

    del model
    
    return emb, lbl


def calculate_cmc(gallery_embedding, probe_embedding, gallery_label, probe_label, last_rank=10):
    """
    :param gallery_embedding: [num of gallery images x embedding size] (n x e) torch float tensor
    :param probe_embedding: [num of probe images x embedding size] (m x e) torch float tensor
    :param gallery_label: [num of gallery images x num of labels] (n x l) torch one hot matrix
    :param label: [num of probe images x num of labels] (m x l) torch one hot matrix
    :param last_rank: the last rank of cmc curve
    :return: (x_range, cmc) where x_range is range of ranks and cmc is probability list with length of last_rank
    """
    gallery_embedding = gallery_embedding.type(torch.float32)
    probe_embedding = probe_embedding.type(torch.float32)
    gallery_label = gallery_label.type(torch.float32)
    probe_label = probe_label.type(torch.float32)


    nof_query = probe_label.shape[0]
    gallery_embedding /= torch.norm(gallery_embedding, p=2, dim=1, keepdim=True)
    probe_embedding /= torch.norm(probe_embedding, p=2, dim=1, keepdim=True)
    prediction_score = torch.matmul(probe_embedding, gallery_embedding.t())
    gt_similarity = torch.matmul(probe_label, gallery_label.t())
    _, sorted_similarity_idx = torch.sort(prediction_score, dim=1, descending=True)
    cmc = torch.zeros(last_rank).type(torch.float32)
    for i in range(nof_query):
        gt_vector = (gt_similarity[i] > 0).type(torch.float32)
        pred_idx = sorted_similarity_idx[i]
        predicted_gt = gt_vector[pred_idx]
        first_gt = torch.nonzero(predicted_gt).type(torch.int)[0]
        if first_gt < last_rank:
            cmc[first_gt:] += 1
    cmc /= nof_query

    if cmc.device.type == 'cuda':
        cmc = cmc.cpu()

    x_range = np.arange(0,last_rank)+1

    return cmc.numpy()


#### Cross-Modal CMC
def cm_cmc_extractor(model, root_pth=config.evaluation['identification'], facenet = None, perinet = None, device = 'cuda:0', rank=10):
    if facenet is None and perinet is None:
        facenet = model
        perinet = model
    total_cmc_f = np.empty((0, rank), int) 
    total_cmc_p = np.empty((0, rank), int) 

    for datasets in dset_list:
        cmc_lst_f = np.empty((0, rank), int)
        cmc_lst_p = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'     
        peri_data_loaders = []
        peri_data_sets = []     
        face_data_loaders = []
        face_data_sets = []            

        # data loader and datasets
        if not datasets in ['ethnic']:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('/')[-1]
                if base_nm == 'gallery':
                    peri_data_load_gal, peri_data_set_gal = data_loader.gen_data(directs + '/peri/', 'test', type='periocular', aug='False')
                    face_data_load_gal, face_data_set_gal = data_loader.gen_data(directs + '/face/', 'test', type='face', aug='False')
                else:
                    peri_data_load, peri_data_set = data_loader.gen_data(directs + '/peri/', 'test', type='periocular', aug='False')
                    face_data_load, face_data_set = data_loader.gen_data(directs + '/face/', 'test', type='face', aug='False')
                    peri_data_loaders.append(peri_data_load)
                    peri_data_sets.append(peri_data_set)
                    face_data_loaders.append(face_data_load)
                    face_data_sets.append(face_data_set)
        # print(datasets)
        # *** ***
        if datasets == 'ethnic':
            p_ethnic_gal_data_load, p_ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', type='periocular', aug='False')
            p_ethnic_pr_data_load, p_ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', type='periocular', aug='False')
            p_ethnic_fea_gal, p_ethnic_lbl_gal = feature_extractor(perinet, p_ethnic_gal_data_load, device = device, peri_flag = True)
            p_ethnic_fea_pr, p_ethnic_lbl_pr = feature_extractor(perinet, p_ethnic_pr_data_load, device = device, peri_flag = True)            
            p_ethnic_lbl_pr, p_ethnic_lbl_gal = F.one_hot(p_ethnic_lbl_pr), F.one_hot(p_ethnic_lbl_gal)

            f_ethnic_gal_data_load, f_ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', type='face', aug='False')
            f_ethnic_pr_data_load, f_ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', type='face', aug='False')
            f_ethnic_fea_gal, f_ethnic_lbl_gal = feature_extractor(facenet, f_ethnic_gal_data_load, device = device, peri_flag = False)
            f_ethnic_fea_pr, f_ethnic_lbl_pr = feature_extractor(facenet, f_ethnic_pr_data_load, device = device, peri_flag = False)            
            f_ethnic_lbl_pr, f_ethnic_lbl_gal = F.one_hot(f_ethnic_lbl_pr), F.one_hot(f_ethnic_lbl_gal)

            cmc_f = calculate_cmc(f_ethnic_fea_gal, p_ethnic_fea_pr, f_ethnic_lbl_gal, p_ethnic_lbl_pr, last_rank=rank)
            cmc_p = calculate_cmc(p_ethnic_fea_gal, f_ethnic_fea_pr, p_ethnic_lbl_gal, f_ethnic_lbl_pr, last_rank=rank)

        else:            
            for probes in peri_data_loaders:                
                face_fea_gal, face_lbl_gal = feature_extractor(perinet, face_data_load_gal, device = device, peri_flag = False)
                peri_fea_pr, peri_lbl_pr = feature_extractor(perinet, probes, device = device, peri_flag = True)
                peri_lbl_pr, face_lbl_gal = F.one_hot(peri_lbl_pr), F.one_hot(face_lbl_gal)

                cmc_f = calculate_cmc(face_fea_gal, peri_fea_pr, face_lbl_gal, peri_lbl_pr, last_rank=rank)
                cmc_lst_f = np.append(cmc_lst_f, np.array([cmc_f]), axis=0)

            for probes in face_data_loaders:                
                peri_fea_gal, peri_lbl_gal = feature_extractor(perinet, peri_data_load_gal, device = device, peri_flag = True)
                face_fea_pr, face_lbl_pr = feature_extractor(perinet, probes, device = device, peri_flag = False)
                face_lbl_pr, peri_lbl_gal = F.one_hot(face_lbl_pr), F.one_hot(peri_lbl_gal)

                cmc_p = calculate_cmc(peri_fea_gal, face_fea_pr, peri_lbl_gal, face_lbl_pr, last_rank=rank)
                cmc_lst_p = np.append(cmc_lst_p, np.array([cmc_p]), axis=0)
                
            cmc_f = np.mean(cmc_lst_f, axis=0)
            cmc_p = np.mean(cmc_lst_p, axis=0)

        cm_cmc_dict_p[datasets] = cmc_p
        cm_cmc_dict_f[datasets] = cmc_f
        # print(datasets)
        # print('Peri Gallery:', cmc_p)        
        # print('Face Gallery:', cmc_f)        

    for ds in cm_cmc_dict_f:
        total_cmc_f = np.append(total_cmc_f, np.array([cm_cmc_dict_f[ds]]), axis=0)
    cm_cmc_avg_dict_f['avg'] = np.mean(total_cmc_f, axis = 0)

    for ds in cm_cmc_dict_p:
        total_cmc_p = np.append(total_cmc_p, np.array([cm_cmc_dict_p[ds]]), axis=0)
    cm_cmc_avg_dict_p['avg'] = np.mean(total_cmc_p, axis = 0)

    return cm_cmc_dict_p, cm_cmc_avg_dict_p, cm_cmc_dict_f, cm_cmc_avg_dict_f


if __name__ == '__main__':
    method = 'AELNet'
    rank = 10  # CMC - rank > 1 (graph) or identification - rank = 1 (values)
    if rank > 1:
        create_folder(method)
    embd_dim = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_model_path = './models/best_model/AELNet.pth'
    model = net.AEL_Net(embedding_size = embd_dim, do_prob=0.0).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)

    cm_cmc_dict_p, cm_avg_dict_p, cm_cmc_dict_f, cm_avg_dict_f = cm_cmc_extractor(model, facenet=None, perinet=None, root_pth=config.evaluation['identification'], device=device, rank=rank)
    if rank > 1:
        cm_cmc_dict_p = copy.deepcopy(cm_cmc_dict_p)
        cm_avg_dict_p = copy.deepcopy(cm_avg_dict_p)
        cm_cmc_dict_f = copy.deepcopy(cm_cmc_dict_f)
        cm_avg_dict_f = copy.deepcopy(cm_avg_dict_f)
        torch.save(cm_cmc_dict_p, './data/cmc/' + str(method) + '/cm/cm_cmc_dict_p.pt')
        torch.save(cm_avg_dict_p, './data/cmc/' + str(method) + '/cm/cm_avg_dict_p.pt')
        torch.save(cm_cmc_dict_f, './data/cmc/' + str(method) + '/cm/cm_cmc_dict_f.pt')
        torch.save(cm_avg_dict_f, './data/cmc/' + str(method) + '/cm/cm_avg_dict_f.pt')    
    cm_cmc_dict_p = get_avg(cm_cmc_dict_p)
    cm_cmc_dict_f = get_avg(cm_cmc_dict_f)    
    print('Average IR (Cross-Modal Periocular): \n', cm_cmc_dict_p['avg'], '±', cm_cmc_dict_p['std'])
    print('Average IR (Cross-Modal Face): \n', cm_cmc_dict_f['avg'], '±', cm_cmc_dict_f['std'])
    