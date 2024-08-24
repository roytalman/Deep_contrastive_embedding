import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_loss():
    
    # loss equation is sum(Cn - Cp + M).
    # Where Cn  is the cosine similarity of negative paires (different labels)
    #       Cp is the cosine similarty of positive paires (same labels)
    #   and M is the mergin 
    
    def __init__(self,margin = 0.4):
        self.margin = margin
        
    def __call__(self,x,labels):
        # x      - n*m embedding matrix where n is batch size and m is embedding size
        # labels - m vector of labels. If you dont have limited number of class, but only positive paires, you can pass each pair with different label.
        
        # calculate n*n cosine similarity matrix 
        cos_sim = F.cosine_similarity(x,x.unsqueeze(1),dim=2)
        # remove diagonal values (self cosine equal to 1 )
        cos_sim_no_diag  = cos_sim.flatten()[:-1].view(len(labels)-1,len(labels)+1)[:,1:].flatten()
        # binary vector eqvivalent to 'cos_sim_no_diag' of same/other label
        labels_equal  = (labels == labels.unsqueeze(1)).flatten()[:-1].view(len(labels)-1,len(labels)+1)[:,1:].flatten()
        
        # find same label paires and different label paires
        pos_cos  = cos_sim_no_diag[labels_equal]
        neg_cos  = cos_sim_no_diag[~labels_equal]
        
        # sort for random positive-negative paires:
        L = min(len(pos_cos),len(neg_cos))
        pos_cos_select =  pos_cos[np.random.choice(len(pos_cos),L, replace=False)] 
        neg_cos_select =  neg_cos[np.random.choice(len(neg_cos),L, replace=False)]
        # calculate embedding pair loss:
        loss_all = torch.max( neg_cos_select - pos_cos_select + self.margin, torch.zeros(L) )
        return loss_all.mean()/L
