import numpy as np
import time
import sys
import itertools 
import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Beta
import math
from sklearn.metrics import pairwise

# **********    

class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x
        self.fn_no_mean = lambda x, i: x

    def __call__(self, loss, loss_psi_1, loss_psi_2, loss_psi_3, loss_psi_4, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:5.4f} | '.format(self.fn(loss, i))
        loss_psi_1 = 'loss_psi_1: {:5.4f} | '.format(self.fn_no_mean(loss_psi_1, i))
        loss_psi_2 = 'loss_psi_2: {:5.4f} | '.format(self.fn_no_mean(loss_psi_2, i))
        loss_psi_3 = 'loss_psi_3: {:5.4f} | '.format(self.fn_no_mean(loss_psi_3, i))
        loss_psi_4 = 'loss_psi_4: {:5.4f} | '.format(self.fn_no_mean(loss_psi_4, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + loss_psi_1 + loss_psi_2 + loss_psi_3 + loss_psi_4 + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')

# **********

class BatchTimer(object):
    
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    
    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)

# **********

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()
    
# **********

def plot_graph(mixed_ST, nrow=8):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import cv2
    def convert_imgs(imgs):
        for i in range(imgs.shape[0]):
            imgs[i] = imgs[i] #torch.from_numpy(cv2.cvtColor(imgs[i].cpu().detach().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB).transpose(2,0,1))
        return imgs

    # show_batch(mixed_query, 32)
    imgs = convert_imgs(mixed_ST)
    grid_img = make_grid(imgs, nrow=nrow).cpu().detach()
    fig=plt.figure()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

# **********

def run_train(model, psi_1, psi_2, psi_3, psi_4,
                face_loader, peri_loader, face_loader_tl, peri_loader_tl, 
                epoch = 1, net_params = None, loss_fn = None, optimizer = None, 
                scheduler = None, batch_metrics = {'time': BatchTimer()}, 
                show_running = True, device = 'cuda:0', writer = None):
    
    mode = 'Train'
    iter_max = len(face_loader)
    logger = Logger(mode, length = iter_max, calculate_mean = show_running)
    
    loss = 0
    metrics = {}
    # wandb.init(project="test")
    
    # **********
    
    face_iterator_tl = iter(face_loader_tl)
    peri_iterator_tl = iter(peri_loader_tl)
    m = Beta(torch.tensor([2.0]), torch.tensor([5.0]))
    wt = math.exp(-5.0 * (1 - (epoch / net_params['epochs'])**2) ) # ramp-up weighting w(t)

    for batch_idx, ( face_in, peri_in ) in enumerate( zip( face_loader, peri_loader ) ):
        #### *** non-target : face ***
        #### random sampling
        face_in, face_aug_in = face_in

        face_x, face_y = face_in
        face_x_aug, face_y_aug = face_aug_in

        face_x = face_x.to(device)
        face_y = face_y.to(device)
        face_x_aug = face_x_aug.to(device)
        face_y_aug = face_y_aug.to(device)

        del face_in, face_aug_in

        #### balanced sampling
        try:
            face_in_tl, face_aug_in_tl = next(face_iterator_tl)
        except StopIteration:
            face_iterator_tl = iter(face_loader_tl)
            face_in_tl, face_aug_in_tl = next(face_iterator_tl)
        
        face_x_tl, face_y_tl = face_in_tl
        face_x_aug_tl, face_y_aug_tl = face_aug_in_tl

        face_x_tl = face_x_tl.to(device)
        face_y_tl = face_y_tl.to(device)
        face_x_aug_tl = face_x_aug_tl.to(device)
        face_y_aug_tl = face_y_aug_tl.to(device)

        del face_in_tl, face_aug_in_tl
        
        # *** ***
        
        face_emb_r = model(torch.cat((face_x, face_x_aug), dim=0))
        face_lbl_r = torch.cat((face_y, face_y_aug))
        face_emb_tl = model(torch.cat((face_x_tl, face_x_aug_tl), dim=0))
        face_lbl_tl = torch.cat((face_y_tl, face_y_aug_tl))    

        del face_x, face_x_aug
        del face_x_tl, face_x_aug_tl

        #
        # *** ***
        #

        #### *** target : periocular ***
        #### random sampling
        peri_in, peri_aug_in = peri_in

        peri_x, peri_y = peri_in
        peri_x_aug, peri_y_aug = peri_aug_in

        peri_x = peri_x.to(device)
        peri_y = peri_y.to(device)
        peri_x_aug = peri_x_aug.to(device)
        peri_y_aug = peri_y_aug.to(device)

        del peri_in, peri_aug_in

        #### balanced sampling
        try:
            peri_in_tl, peri_aug_in_tl = next(peri_iterator_tl)
        except StopIteration:
            peri_iterator_tl = iter(peri_loader_tl)
            peri_in_tl, peri_aug_in_tl = next(peri_iterator_tl)
        
        peri_x_tl, peri_y_tl = peri_in_tl
        peri_x_aug_tl, peri_y_aug_tl = peri_aug_in_tl

        peri_x_tl = peri_x_tl.to(device)
        peri_y_tl = peri_y_tl.to(device)
        peri_x_aug_tl = peri_x_aug_tl.to(device)
        peri_y_aug_tl = peri_y_aug_tl.to(device)

        del peri_in_tl, peri_aug_in_tl

        # *** ***

        peri_emb_r  = model(torch.cat((peri_x, peri_x_aug), dim=0))
        peri_lbl_r = torch.cat((peri_y, peri_y_aug))
        peri_emb_tl = model(torch.cat((peri_x_tl, peri_x_aug_tl), dim=0))
        peri_lbl_tl = torch.cat((peri_y_tl, peri_y_aug_tl))

        del peri_x, peri_x_aug
        del peri_x_tl, peri_x_aug_tl

        # *** ***

        assert (torch.all(peri_lbl_tl == face_lbl_tl))

        emb_r = torch.cat((peri_emb_r, face_emb_r), dim=0)
        lbl_r = torch.cat((peri_lbl_r, face_lbl_r))
        emb_tl = torch.cat((peri_emb_tl, face_emb_tl), dim=0)
        lbl_tl = torch.cat((peri_lbl_tl, face_lbl_tl))
       
        # *** ***         
        # classifier head 1 - random periocular-face embeddings
        pred_1 = psi_1(emb_r, lbl_r)
        loss_psi_1 = loss_fn['loss_ce'](pred_1, lbl_r)

        # ***
        # classifier head 2 - balanced periocular-face embedding pairs
        pred_2 = psi_2(emb_tl, lbl_tl)
        loss_psi_2 = loss_fn['loss_ce'](pred_2, lbl_tl)

        # ***
        # classifier head 3 - periocular-face embedding level mean
        cross_emb_mean = torch.cat((peri_emb_tl.unsqueeze(1), face_emb_tl.unsqueeze(1)), dim=1)
        cross_emb_mean = torch.mean(cross_emb_mean, dim=1)
        mean_pred_3 = psi_3(cross_emb_mean, peri_lbl_tl)
        loss_psi_3 = loss_fn['loss_ce'](mean_pred_3, peri_lbl_tl)

        # ***
        # classifier head 4 - periocular-face embedding and classifier level mixup
        beta_1 = m.sample().to(device)
        face_pred_4 = psi_4(face_emb_tl, face_lbl_tl)
        peri_pred_4 = psi_4(peri_emb_tl, peri_lbl_tl)
        pred_4_mixup = beta_1 * face_pred_4 + (1 - beta_1) * peri_pred_4
        pred_loss_psi_4 = loss_fn['loss_ce'](pred_4_mixup, face_lbl_tl)

        beta_2 = m.sample().to(device)
        cross_emb_mixup = beta_2 * face_emb_tl + (1 - beta_2) * peri_emb_tl
        mixup_pred_4 = psi_4(cross_emb_mixup, peri_lbl_tl)
        emb_loss_psi_4 = loss_fn['loss_ce'](mixup_pred_4, peri_lbl_tl)
        
        # *** ***
                    
        del face_emb_tl, peri_emb_tl, face_emb_r, peri_emb_r, emb_tl, emb_r
        
        # *** ***
                        
        # Define loss_batch
        loss_batch = (net_params['weight_1'] * loss_psi_1) + (net_params['weight_2'] * loss_psi_2) + (wt * (loss_psi_3)) + (wt * (pred_loss_psi_4 + emb_loss_psi_4))

        # *** ***

        # if model.training:
        optimizer.zero_grad()
        loss_batch.backward() 
        optimizer.step()
        
        time.sleep(0.0001)
        
        # *** ***
        
        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(pred_1, lbl_r).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
            
        if writer is not None: # and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, (net_params['weight_1'] * loss_psi_1), (net_params['weight_2'] * loss_psi_2), (wt * (loss_psi_3)), (wt * (pred_loss_psi_4 + emb_loss_psi_4)), metrics, batch_idx)
        else:
            logger(loss_batch, metrics_batch, batch_idx)
            
    # END FOR
    # Completed processing for all batches (training and testing), i.e., an epoch 
    
    # *** ***
        
    # if model.training and scheduler is not None:
    if scheduler is not None:
        scheduler.step()

    loss = loss / (batch_idx + 1)
    metrics = {k: v / (batch_idx + 1) for k, v in metrics.items()}
    
    return metrics, loss

# **********

def run_test(model, gallery_loader, probe_loader, device = 'cuda:0', peri_flag = False, proto_flag = False):
    model = model.eval()

    gal_acc, pr_acc = validate_identification(model, gallery_loader, probe_loader, device = device, peri_flag = peri_flag, proto_flag = proto_flag)

    return gal_acc, pr_acc

# **********

def validate_identification(model, loader_gallery, loader_test, device = 'cuda:0', peri_flag = False, proto_flag = False):
    
    # ***** *****
    
    # model = model.eval().to(device)
    # model.classify = False
        
    # ***** *****
    
    # Extract gallery features w.r.t. pre-learned model
    gallery_fea = torch.tensor([])
    gallery_label = torch.tensor([], dtype = torch.int16)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_gallery):

            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

            gallery_fea = torch.cat((gallery_fea, x.detach().cpu()), 0)
            gallery_label = torch.cat((gallery_label, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(gallery_fea.size()[0] == gallery_label.size()[0])
    
    del loader_gallery
    time.sleep(0.0001)
    
    # ***** *****
    
    # Extract test features w.r.t. pre-learned model
    test_fea = torch.tensor([])
    test_label = torch.tensor([], dtype = torch.int16)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_test):

            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

            test_fea = torch.cat((test_fea, x.detach().cpu()), 0)
            test_label = torch.cat((test_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    # print('Test Set Capacity\t: ', test_fea.size())
    assert(test_fea.size()[0] == test_label.size()[0])
    
    del loader_test
    time.sleep(0.0001)

    # ***** *****

    # prototyping for gallery
    if proto_flag is True:
        gal_lbl_proto = torch.tensor([], dtype = torch.int64)
        gal_fea_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(gallery_label):
            # append unique labels to tensor list
            gal_lbl_proto = torch.cat((gal_lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            indices = np.where(gallery_label == i)
            gal_feats = torch.tensor([])

            # from index list, append features into temporary gal_feats list
            for j in indices:
                gal_feats = torch.cat((gal_feats, gallery_fea[j].detach().cpu()), 0)
            # print(gal_feats.shape)
            # get mean of full gal_feats list, and then unsqueeze to append the average prototype value into gal_fea_proto
            proto_mean = torch.unsqueeze(torch.mean(gal_feats, 0), 0)
            proto_mean = F.normalize(proto_mean, p=2, dim=1)
            gal_fea_proto = torch.cat((gal_fea_proto, proto_mean.detach().cpu()), 0)
    
        gallery_fea, gallery_label = gal_fea_proto, gal_lbl_proto
    
    # Calculate gallery_acc and test_acc
    gallery_label = np.reshape(np.array(gallery_label), -1)
    test_label = np.reshape(np.array(test_label), -1)
    
    gallery_dist = pairwise.cosine_similarity(gallery_fea)
    gallery_pred = np.argmax(gallery_dist, 0)
    gallery_pred = gallery_label[gallery_pred] 
    gallery_acc = sum(gallery_label == gallery_pred) / gallery_label.shape[0]
    
    test_dist = pairwise.cosine_similarity(gallery_fea, test_fea)
    test_pred = np.argmax(test_dist, 0)
    test_pred = gallery_label[test_pred]
    test_acc = sum(test_label == test_pred) / test_label.shape[0]

    # torch.cuda.empty_cache()
    # time.sleep(0.0001)
    
    del model
    time.sleep(0.0001)
    
    return gallery_acc, test_acc

# *****    

def feature_extractor(model, data_loader, device = 'cuda:0', peri_flag = False, proto_flag = False):    
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

    if proto_flag is True:
        lbl_proto = torch.tensor([], dtype = torch.int64)
        emb_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(lbl):
            # append unique labels to tensor list
            lbl_proto = torch.cat((lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            indices = np.where(lbl == i)
            feats = torch.tensor([])

            # from index list, append features into temporary feats list
            for j in indices:
                feats = torch.cat((feats, emb[j].detach().cpu()), 0)
            # print(feats.shape)
            # get mean of full feats list, and then unsqueeze to append the average prototype value into gal_fea_proto
            proto_mean = torch.unsqueeze(torch.mean(feats, 0), 0)
            proto_mean = F.normalize(proto_mean, p=2, dim=1)
            emb_proto = torch.cat((emb_proto, proto_mean.detach().cpu()), 0)
    
        emb, lbl = emb_proto, lbl_proto

    # print('Set Capacity\t: ', emb.size())
    assert(emb.size()[0] == lbl.size()[0])
    
    del data_loader
    time.sleep(0.0001)

    del model
    
    return emb, lbl


# **********