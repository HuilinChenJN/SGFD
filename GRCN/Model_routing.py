import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from SAGEConv import SAGEConv
from GATConv import GATConv
from torch_geometric.utils import add_self_loops, dropout_adj
from torch_scatter import scatter_mean
# from torch.utils.checkpoint import checkpoint
from feature_extractor.FeatureExtractorModel import FeatureExtractorModel
from feature_fusion.FeatureFusionModel import FeatureFusionModel


##########################################################################


def recall_at_k(r, k, all_poss_num):
    r = list(r)
    r = np.asfarray(r)[:k]
    return len(r) / all_poss_num


def hit_at_k(r, k):
    r = list(r)
    r = np.asfarray(r)[:k]
    if len(r) > 0:
        return 1.
    else:
        return 0.


class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        if self.has_norm:
            x = F.normalize(x)

        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)

        return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(self, features, meta_label, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm,
                 t_decay, is_word=False):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.meta_label = meta_label
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)
        self.is_word = is_word
        self.transfer_feature = None
        if is_word:
            self.word_tensor = torch.LongTensor(features).cuda()
            self.features = nn.Embedding(torch.max(features[1]) + 1, dim_C)
            nn.init.xavier_normal_(self.features.weight)

        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.feature_extractor_model = FeatureExtractorModel(self.features, self.dim_feat, self.meta_label,
                                                                 self.dim_C, t_decay, self.has_act)

    def forward(self, edge_index, item_tensor):

        features = None

        if self.is_word:
            features = scatter_mean(self.features(self.word_tensor[1]), self.word_tensor[0])
            # features = torch.tensor(scatter_('mean', self.features(self.word_tensor[1]), self.word_tensor[0])).cuda()
        else:
            segment_num = 50
            segment_len = int(self.features.size(0) / segment_num)
            for index in range(segment_num):
                segement = torch.tensor(index * segment_len, dtype=torch.int64)
                net_segment = torch.tensor((index + 1) * segment_len, dtype=torch.int64)
                if index == (segment_num - 1):
                    temp_features = self.features[segement:]
                else:
                    temp_features = self.features[segement:net_segment]
                # 特征提取
                transferred_temp_features = self.feature_extractor_model.student_model(temp_features)

                if features is None:
                    features = transferred_temp_features
                else:
                    features = torch.cat([features, transferred_temp_features], dim=0)
            self.transfer_feature = F.leaky_relu(features)
            features = F.leaky_relu(features)

        if self.has_norm:
            preference = F.normalize(self.preference)
            features = F.normalize(features)

        for i in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            x_hat_1 = self.conv_embed_1(x, edge_index)
            preference = preference + x_hat_1[:self.num_user]

            if self.has_norm:
                preference = F.normalize(preference)

        x = torch.cat((preference, features), dim=0)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        x_hat_1 = self.conv_embed_1(x, edge_index)

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        item_nodes = item_tensor - self.num_user
        tran_feature, class_loss, kd_loss, feature_loss = self.feature_extractor_model(item_nodes)

        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1), class_loss, kd_loss, feature_loss, tran_feature


class Net(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, reg_weight,
                 ce_weight, kd_weight, t_decay,
                 v_feat, a_feat, t_feat, meta_data,
                 aggr_mode, weight_mode, fusion_mode,
                 num_routing, dropout,
                 has_act, has_norm, has_entropy_loss, has_weight_loss,
                 is_word,
                 dim_E, dim_C,
                 pruning):
        super(Net, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.weight_mode = weight_mode
        self.fusion_mode = fusion_mode
        self.weight = torch.tensor([[1.0], [-1.0]]).cuda()
        self.reg_weight = reg_weight
        self.dropout = dropout

        # kd loss
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight
        self.t_decay = t_decay

        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.id_gcn = EGCN(num_user, num_item, dim_E, aggr_mode, has_act, has_norm)
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat
        self.meta_data = torch.tensor(meta_data, dtype=torch.int64).cuda()
        self.has_entropy_loss = has_entropy_loss
        self.has_weight_loss = has_weight_loss

        self.v_rep = None
        self.a_rep = None
        self.t_rep = None


        self.pruning = pruning

        num_model = 0
        if v_feat is not None:
            self.v_gcn = CGCN(self.v_feat, self.meta_data, num_user, num_item, dim_C, aggr_mode, num_routing, has_act,
                              has_norm, self.t_decay)
            num_model += 1

        if a_feat is not None:
            self.a_gcn = CGCN(self.a_feat, self.meta_data, num_user, num_item, dim_C, aggr_mode, num_routing, has_act,
                              has_norm, self.t_decay)
            num_model += 1

        if t_feat is not None:
            self.t_gcn = CGCN(self.t_feat, self.meta_data, num_user, num_item, dim_C, aggr_mode, num_routing, has_act,
                              has_norm, self.t_decay, is_word)
            num_model += 1

        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, num_model))))

        self.feature_fusion_model = FeatureFusionModel(self.meta_data, dim_C)

        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))).cuda()

    def forward(self, item_tensor):
        weight = None
        content_rep = None
        num_modal = 0
        class_loss = None
        kd_loss = None
        feature_loss = None
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v, class_loss_v, kd_loss_v, feature_loss_v, student_v = self.v_gcn(edge_index, item_tensor)
            weight = weight_v
            content_rep = v_rep
            self.v_rep = v_rep
            class_loss = class_loss_v
            kd_loss = kd_loss_v
            feature_loss = feature_loss_v

        if self.a_feat is not None:
            num_modal += 1
            a_rep, weight_a, class_loss_a, kd_loss_a, feature_loss_a, student_a = self.a_gcn(edge_index, item_tensor)
            self.a_rep = a_rep
            if weight is None:
                weight = weight_a
                content_rep = a_rep
                class_loss = class_loss_a
                kd_loss = kd_loss_a
                feature_loss = feature_loss_a
            else:
                class_loss = class_loss + class_loss_a
                kd_loss = kd_loss + kd_loss_a
                feature_loss = feature_loss + feature_loss_a
                content_rep = torch.cat((content_rep, a_rep), dim=1)
                if self.weight_mode == 'mean':
                    weight = weight + weight_a
                else:
                    weight = torch.cat((weight, weight_a), dim=1)

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t, class_loss_t, kd_loss_t, feature_loss_t, student_t = self.t_gcn(edge_index, item_tensor)
            self.t_rep = t_rep
            if weight is None:
                weight = weight_t
                conetent_rep = t_rep
                class_loss = class_loss_t
                kd_loss = kd_loss_t
                feature_loss = feature_loss_t
            else:
                class_loss = class_loss + class_loss_t
                kd_loss = kd_loss + kd_loss_t
                feature_loss = feature_loss + feature_loss_t
                content_rep = torch.cat((content_rep, t_rep), dim=1)
                if self.weight_mode == 'mean':
                    weight = weight + weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)

        if self.weight_mode == 'mean':
            weight = weight / num_modal

        elif self.weight_mode == 'max':
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]),
                                   dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        if self.pruning:
            weight = torch.relu(weight)

        id_rep = self.id_gcn(edge_index, weight)

        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            representation = (id_rep + v_rep + a_rep + t_rep) / 4

        self.result = representation

        fused_feature = torch.cat([student_v, student_t], dim=1)
        item_nodes = item_tensor - self.num_user
        fusion_loss = self.feature_fusion_model(item_nodes, fused_feature)

        class_loss = class_loss / num_modal
        kd_loss = kd_loss / num_modal
        feature_loss = feature_loss / num_modal



        return representation, (fusion_loss + class_loss), (kd_loss + feature_loss)

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out, classfiy_loss, kd_loss = self.forward(item_tensor)
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))

        extra_feature_loss = self.ce_weight * classfiy_loss + self.kd_weight * kd_loss +  bpr_loss

        reg_embedding_loss = (
                    self.id_gcn.id_embedding[user_tensor] ** 2 + self.id_gcn.id_embedding[item_tensor] ** 2).mean()
        reg_content_loss = torch.zeros(1).cuda()
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor] ** 2).mean()
        if self.a_feat is not None:
            reg_content_loss = reg_content_loss + (self.a_gcn.preference[user_tensor] ** 2).mean()
        if self.t_feat is not None:
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor] ** 2).mean()

        reg_confid_loss = (self.model_specific_conf ** 2).mean()

        reg_loss = reg_embedding_loss + reg_content_loss

        reg_loss = self.reg_weight * reg_loss

        loss = extra_feature_loss + reg_loss

        return loss, bpr_loss, reg_confid_loss, reg_embedding_loss + reg_content_loss, \
               reg_embedding_loss, extra_feature_loss

    def accuracy(self, step=2000, topk=10):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step == None else step
        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu() + self.num_user),
                                               dim=0)
            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = self.num_user
        precision = recall = ndcg = hk_recall = hk = 0.0

        for row, col in self.user_item_dict.items():
            user = row
            pos_items = set(col)
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)
            hit_items = pos_items.intersection(items)
            num_hit = len(hit_items)

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)
            hk_recall += recall_at_k(hit_items, topk, num_pos)
            hk += hit_at_k(hit_items, topk)
            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / (np.log(i + 2))
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / (np.log(i + 2))

            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length, hk / length, hk_recall / length

    def full_accuracy(self, val_data, step=2000, topk=20):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - self.num_user
                    score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu() + self.num_user),
                                               dim=0)
            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = 0  # len(val_data)
        precision = recall = ndcg = hk_recall = hk = 0.0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            # 获取当前用户的 推荐序列和推荐值
            items_list = all_index_of_rank_list[user].tolist()
            # print(user, pos_items, items_list)
            items = set(items_list)
            hit_items = pos_items.intersection(items)
            num_hit = len(hit_items)

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)
            hk_recall += recall_at_k(hit_items, topk, num_pos)
            hk += hit_at_k(hit_items, topk)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log(i + 2)
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log(i + 2)

            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length, hk / length, hk_recall / length
