import torch
import torch.nn as nn
import torch.nn.functional as F
from .TeacherStudentModel import TeacherModel,StudentModel


class FeatureExtractorModel(torch.nn.Module):
    def __init__(self, feature, feature_dim, meta_label, dim_latent=64, t=100, is_pruning=True):
        super(FeatureExtractorModel, self).__init__()
        self.meta_label = meta_label
        self.label_size = len(set(meta_label.tolist()))
        self.dim_latent = dim_latent
        self.is_pruning = is_pruning
        self.features = feature
        self.t = t
        self.kd_weight = []
        # MLP classifier
        self.category_classification = nn.Linear(self.dim_latent, self.label_size)

        self.teacher_model = TeacherModel(feature_dim, self.dim_latent,  self.is_pruning)
        self.student_model = StudentModel(feature_dim, self.dim_latent,  self.is_pruning)

    def forward(self, nodes):
        node_feature = self.features[nodes]
        node_label = self.meta_label[nodes]
        # 1. obtain the probability distribution of teacher model
        teacher_x = self.teacher_model(node_feature)
        teacher_result = self.category_classification(teacher_x)
        teacher_soft_result = F.softmax(teacher_result / self.t, dim=-1)
        teacher_hard_result = F.softmax(teacher_result, dim=-1)

        # 2. obtain the probability distribution of student model
        student_x = self.student_model(node_feature)
        student_result = self.category_classification(student_x)
        student_soft_result = F.softmax(student_result / self.t, dim=-1)

        # 3. The teacher extractor is optimized using CE loss
        label_class_loss = F.cross_entropy(teacher_hard_result, node_label)

        # 4. obtain the knowledge distillation based on response and feature
        label_kd_loss = F.l1_loss(teacher_soft_result, student_soft_result)
        feature_constraint_loss = F.mse_loss(student_x, teacher_x)

        return teacher_x, label_class_loss, label_kd_loss, feature_constraint_loss
