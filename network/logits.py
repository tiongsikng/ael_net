import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import os, sys
import math
import sys
import random
import numpy as np

# *** *** *** *** ***
# ***** CrossEntropy *****

class CrossEntropy(nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super(CrossEntropy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
   
    def forward(self, x):
        
        x = F.linear(F.normalize(x, p=2, dim=1), self.weight)
        # x = F.linear(F.normalize(x, p=2, dim=1), \
        #             F.normalize(self.weight, p=2, dim=1))

        return x
       
    def __repr__(self):

        return self.__class__.__name__ + '(' \
           + 'in_features = ' + str(self.in_features) \
           + ', out_features = ' + str(self.out_features) + ')'
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


# **************** 

# URL-1 : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# URL-2 : https://github.com/MuggleWang/CosFace_pytorch/blob/master/main.py
#    Args:
#        in_features: size of each input sample
#        out_features: size of each output sample
#        s: norm of input feature
#        m: margin

class CosFace(nn.Module):
    
    def __init__(self, in_features, out_features, s = 30.0, m = 0.40):
        
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def forward(self, input, label = None):
        
        # cosine = self.cosine_sim(input, self.weight).clamp(-1,1)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)
        
        return output# , F.normalize(self.weight, p=2, dim=1), (cosine * one_hot)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
    
    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

# ****************

# URL : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# Args:
#    in_features: size of each input sample
#    out_features: size of each output sample
#    s: norm of input feature
#    m: margin
#    cos(theta + m)

class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device = 'cuda:0'):
        
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # ***
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)) 
        self.reset_parameters() 
        self.device = device
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # nan issues: https://github.com/ronghuaiyang/arcface-pytorch/issues/32
        # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        
        return output #, self.weight[label,:]
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

# ****************   
#### LargeMargin and CPFC losses for FPCI

class LargeMarginSoftmax(nn.CrossEntropyLoss):
    """
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    regularization proposed in
       T. Kobayashi, "Large-Margin In Softmax Cross-Entropy Loss." In BMVC2019.
    """

    def __init__(self, reg_lambda=0.3, deg_logit=None,
                 weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginSoftmax, self).__init__(weight=weight, size_average=size_average,
                                                 ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0)  # number of samples
        C = input.size(1)  # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N), target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (C - 1)) * F.log_softmax(X, dim=1) * (1.0 - Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


class total_LargeMargin_CrossEntropy(nn.Module):
    def __init__(self):
        super(total_LargeMargin_CrossEntropy, self).__init__()
        self.loss1 = LargeMarginSoftmax()
        self.loss2 = LargeMarginSoftmax()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s1_loss = self.loss1(s1, target)
        s2_loss = self.loss2(s2, target)

        total_loss = s1_loss + s2_loss

        return total_loss


class CFPC_loss(nn.Module):
    """
    This combines the CrossCLR Loss proposed in
       M. Zolfaghari et al., "CrossCLR: Cross-modal Contrastive Learning For Multi-modal Video Representations,"
       In ICCV2021.
    """

    def __init__(self, temperature=0.02, negative_weight=0.8, device='cuda:0'):
        super(CFPC_loss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.device = device
        self.negative_w = negative_weight  # Weight of negative samples logits.

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask)
        return mask.to(self.device)

    def forward(self, face_features, ocular_features):
        """
        Inputs shape (batch, embed_dim)
        Args:
            face_features: face embeddings (batch, embed_dim)
            ocular_features: ocular embeddings (batch, embed_dim)
        Returns:
        """
        batch_size = face_features.shape[0]

        # Normalize features
        # face_features = nn.functional.normalize(face_features, dim=1)
        # ocular_features = nn.functional.normalize(ocular_features, dim=1)

        # Inter-modality alignment
        logits_per_face = face_features @ ocular_features.t()
        logits_per_ocular = ocular_features @ face_features.t()

        # Intra-modality alignment
        logits_clstr_face = face_features @ face_features.t()
        logits_clstr_ocular = ocular_features @ ocular_features.t()

        logits_per_face /= self.temperature
        logits_per_ocular /= self.temperature
        logits_clstr_face /= self.temperature
        logits_clstr_ocular /= self.temperature

        positive_mask = self._get_positive_mask(face_features.shape[0])
        negatives_face = logits_clstr_face * positive_mask
        negatives_ocular = logits_clstr_ocular * positive_mask

        face_logits = torch.cat([logits_per_face, self.negative_w * negatives_face], dim=1)
        ocular_logits = torch.cat([logits_per_ocular, self.negative_w * negatives_ocular], dim=1)

        diag = np.eye(batch_size)
        mask_face = torch.from_numpy(diag).to(self.device)
        mask_ocular = torch.from_numpy(diag).to(self.device)

        mask_neg_f = torch.zeros_like(negatives_face)
        mask_neg_o = torch.zeros_like(negatives_ocular)
        mask_f = torch.cat([mask_face, mask_neg_f], dim=1)
        mask_o = torch.cat([mask_ocular, mask_neg_o], dim=1)

        loss_f = self.compute_loss(face_logits, mask_f)
        loss_o = self.compute_loss(ocular_logits, mask_o)

        return (loss_f.mean() + loss_o.mean()) / 2