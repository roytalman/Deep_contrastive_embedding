import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_loss():
    
    def __init__(self,margin = 0.4):
        self.margin = margin
        
    def __call__(self,x,labels):
        # x      - n*m matrix where n is barch size and m is embedding size
        # labels - m vector of labels
        
        # calculate n*n cosine similarity matrix 
        cos_sim = F.cosine_similarity(x,x.unsqueeze(1),dim=2)
        # remove diagonal values (self cosine equal to 1 )
        cos_sim_no_diag  = cos_sim.flatten()[:-1].view(len(labels)-1,len(labels)+1)[:,1:].flatten()
        # vector eqvivalent to 'cos_sim_no_diag' of same label or different label
        labels_equal  = (labels == labels.unsqueeze(1)).flatten()[:-1].view(len(labels)-1,len(labels)+1)[:,1:].flatten()
        
        # find same label cosine and different label cosine
        pos_cos  = cos_sim_no_diag[labels_equal]
        neg_cos  = cos_sim_no_diag[~labels_equal]
        
        # sort for random positive-negative paires:
        L = min(len(pos_cos),len(neg_cos))
        pos_cos_select =  pos_cos[np.random.choice(len(pos_cos),L, replace=False)] 
        neg_cos_select =  neg_cos[np.random.choice(len(neg_cos),L, replace=False)]
        # calculate embedding pair loss:
        loss_all = torch.max( neg_cos_select - pos_cos_select + self.margin, torch.zeros(L) )
        return loss_all.mean()