import os, sys, glob, copy
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import time
import torch
import torch.utils.data
from torch.nn import functional as F
from data import data_loader
from sklearn.metrics import pairwise
import network.ael_net as net
from network import load_model
from configs import datasets_config as config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

id_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_id_dict_f = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_id_dict_p = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar'] 


def get_avg(dict_list):
    total_ir = 0
    ir_list = []
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    if 'std' in dict_list.keys():
        del dict_list['std']
    for items in dict_list:
        total_ir += dict_list[items]
    dict_list['avg'] = total_ir/len(dict_list)
    dict_list['std'] = np.std(np.array(ir_list)) * 100

    return dict_list


#### Intra-Modal Identification (Main)
def im_id_main(model, peri_flag=True, root_pth=config.evaluation['identification'], device='cuda:0'):
    modal = 'peri' if peri_flag == True else 'face'
    for datasets in dset_list:
        root_drt = root_pth + datasets + '/**'        
        modal_root = '/' + modal[:4] + '/'
        probe_data_loaders = []
        probe_data_sets = []
        acc = []        

        # data loader and datasets
        for directs in glob.glob(root_drt):
            base_nm = directs.split('\\')[-1]
            modal_base = directs + modal_root
            if not datasets in ['ethnic']:
                if modal_base.split('/')[-3] != 'gallery':
                    data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                    probe_data_loaders.append(data_load)
                    probe_data_sets.append(data_set)
                else: 
                    data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                    gallery_data_loaders, gallery_data_sets = data_load, data_set

        # *** ***

        if datasets == 'ethnic':
            ethnic_gal_data_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            ethnic_pr_data_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            acc = intramodal_id(model, ethnic_gal_data_load, ethnic_pr_data_load, device=device, peri_flag=peri_flag)
        else:
                for i in range(len(probe_data_loaders)):
                    test_acc = intramodal_id(model, gallery_data_loaders, probe_data_loaders[i], device=device, peri_flag=peri_flag)
                    test_acc = np.around(test_acc, 4)
                    acc.append(test_acc)

        # *** ***

        acc = np.around(np.mean(acc), 4)
        print(datasets, acc)
        id_dict[datasets] = acc

    return id_dict


#### Intra-Modal Identification
def intramodal_id(model, gal_loader, probe_loader, device='cuda:0', peri_flag=True):
    
    # ***** *****    
    model = model.eval().to(device)
    # ***** *****

    gal_fea = torch.tensor([])
    gal_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(gal_loader):
            
            x = x.to(device)
            x = model(x, peri_flag=peri_flag)

            gal_fea = torch.cat((gal_fea, x.detach().cpu()), 0)
            gal_label = torch.cat((gal_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    assert(gal_fea.size()[0] == gal_label.size()[0])
    
    del gal_loader
    time.sleep(0.0001)

    # *****    
    probe_fea = torch.tensor([])
    probe_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(probe_loader):

            x = x.to(device)
            x = model(x, peri_flag=peri_flag)

            probe_fea = torch.cat((probe_fea, x.detach().cpu()), 0)
            probe_label = torch.cat((probe_label, y))
            
            del x, y
            time.sleep(0.0001)

    assert(probe_fea.size()[0] == probe_label.size()[0])
    
    del probe_loader
    time.sleep(0.0001)
    
    # ***** *****
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


if __name__ == '__main__':
    embd_dim = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    load_model_path = './models/sota/AELNet.pth'
    model = net.AEL_Net(embedding_size = embd_dim, do_prob=0.0).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)

    peri_id_dict = im_id_main(model, peri_flag=True, root_pth=config.evaluation['identification'], device=device)
    peri_id_dict = get_avg(peri_id_dict)
    peri_id_dict = copy.deepcopy(peri_id_dict)
    print('Average IR (Intra-Modal Periocular): \n', peri_id_dict['avg'], '±', peri_id_dict['std'])

    face_id_dict = im_id_main(model, peri_flag=True, root_pth=config.evaluation['identification'], device=device)
    face_id_dict = get_avg(face_id_dict)
    face_id_dict = copy.deepcopy(face_id_dict)
    print('Average IR (Intra-Modal Face): \n', face_id_dict['avg'], '±', face_id_dict['std'])

    cm_id_dict_p, cm_id_dict_f = cm_id_main(model, root_pth=config.evaluation['identification'], face_model=None, peri_model=None, device=device)
    cm_id_dict_p, cm_id_dict_f = get_avg(cm_id_dict_p), get_avg(cm_id_dict_f)
    cm_id_dict_p = copy.deepcopy(cm_id_dict_p)
    cm_id_dict_f = copy.deepcopy(cm_id_dict_f)
    print('Average IR (Cross-Modal Periocular): \n', cm_id_dict_p['avg'], '±', cm_id_dict_p['std'])
    print('Average IR (Cross-Modal Face): \n', cm_id_dict_f['avg'], '±', cm_id_dict_f['std'])
    