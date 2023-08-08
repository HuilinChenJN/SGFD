import math
# from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(torch.nn.Module):
    def __init__(self, feature_dim, dim_latent, is_pruning=True):
        super(TeacherModel, self).__init__()
        self.feature_dim = feature_dim
        self.dim_latent = dim_latent
        self.decay_ration = 4
        self.is_pruning = is_pruning
        # DNN-based feature extractor in teach model
        liner_feat_dim = self.feature_dim
        linear_list = []
        while True:
            liner_feat_dim = int(liner_feat_dim / self.decay_ration)
            if liner_feat_dim < self.dim_latent:
                linear_list.append(nn.Linear(liner_feat_dim * self.decay_ration, self.dim_latent))
                break
            else:
                linear_list.append(nn.Linear(liner_feat_dim * self.decay_ration, liner_feat_dim))
                if liner_feat_dim == self.dim_latent:
                    break
        self.transfer_multilayer = nn.Sequential(*linear_list)

    def forward(self, x):
        transfer_x = self.transfer_multilayer(x)
        if self.is_pruning:
            transfer_x = F.leaky_relu(transfer_x)

        transfer_x = F.normalize(transfer_x)

        return transfer_x


class StudentModel(torch.nn.Module):
    def __init__(self, feature_dim, dim_latent, is_pruning=True):
        super(StudentModel, self).__init__()
        self.dim_latent = dim_latent
        self.feature_dim = feature_dim
        self.is_pruning = is_pruning
        # SNN-based feature extractor in student model
        self.MLP = nn.Linear(self.feature_dim, self.dim_latent)

    def forward(self, x):
        transfer_x = self.MLP(x)
        if self.is_pruning:
            transfer_x = F.leaky_relu(transfer_x)

        transfer_x = F.normalize(transfer_x)

        return transfer_x
