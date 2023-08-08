import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionModel(torch.nn.Module):
    def __init__(self, meta_label, dim_latent):
        super(FeatureFusionModel, self).__init__()
        self.dim_latent = dim_latent
        self.meta_label = meta_label
        self.label_size = len(set(meta_label.tolist()))
        self.feature_transfer_layer = nn.Linear(2 * dim_latent, dim_latent)
        self.category_classification = nn.Linear(self.dim_latent, self.label_size)

    def forward(self, nodes, fusion, has_n=True):
        gt_prediction = self.meta_label[nodes]
        # fusing textual and visual features
        transfer_layer = self.feature_transfer_layer(fusion)

        if has_n:
            transfer_layer = F.leaky_relu_(transfer_layer)

        result = self.category_classification(transfer_layer)
        class_loss = F.cross_entropy(result, gt_prediction)

        return class_loss

