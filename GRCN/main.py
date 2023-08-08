import argparse
import os
import random
import time
import pandas as pd
from torch import autograd
import numpy as np
import torch
from Dataset import TrainingDataset, VTDataset, data_load
from Model_routing import Net
from torch.utils.data import DataLoader
from Train import train
from Full_t import full_t
from Full_vt import full_vt
# from torch.utils.tensorboard import SummaryWriter
###############################248###########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')

    parser.add_argument('--ce_weight', type=float, default=1e-0, help='Learning rate.')
    parser.add_argument('--kd_weight', type=float, default=1e-0, help='Learning rate.')
    parser.add_argument('--t_decay', type=int, default=100, help='Learning rate.')

    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation Batch size.')

    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--num_routing', type=int, default=3, help='Layer number.')

    parser.add_argument('--dim_E', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--dim_C', type=int, default=128, help='Latent dimension.')

    parser.add_argument('--dropout', type=float, default=0, help='dropout.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation Mode.')
    parser.add_argument('--topK', type=int, default=20, help='Workers number.')

    parser.add_argument('--has_act', default='False', help='Has non-linear function.')
    parser.add_argument('--has_norm', default='True', help='Normalize.')
    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='True', help='Has Visual Features.')
    parser.add_argument('--has_a', default='True', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')

    parser.add_argument('--is_pruning', default='False', help='Pruning Mode')
    parser.add_argument('--weight_mode', default='confid', help='Weight mode')
    parser.add_argument('--fusion_mode', default='concat', help='Fusion mode')
    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    save_file = args.save_file

    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_routing = args.num_routing
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode
    dropout = args.dropout
    weight_mode = args.weight_mode
    fusion_mode = args.fusion_mode
    has_act = True if args.has_act == 'True' else False
    pruning = True if args.is_pruning == 'True' else False
    has_norm = True if args.has_norm == 'True' else False
    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False
    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E
    dim_C = None if args.dim_C == 0 else args.dim_C
    is_word = True if data_path == 'Tiktok' else False
    # kd loss
    # kd_class_decay = eval(args.kd_class_decay)
    ce_weight = args.ce_weight
    kd_weight = args.kd_weight
    t_decay = args.t_decay
    import time

    timeStr = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    writer = None
    # writer = SummaryWriter(os.path.join('./log', args.data_path, timeStr))
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, weight_decay), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t Weight_decay:{1} ---------------------------------\r\n'.format(l_r, weight_decay))
    #########################################################################################################################################
    print('Data loading ...')
    num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(data_path)
    valid_ids = torch.randint(low=0, high= num_item+num_user, size=(1, 200))
    valid_ids = valid_ids.reshape(-1)

    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)#, num_workers=num_workers)

    # 加载meta信息，包括item的分类信息
    meta_data = np.load('../Data/' + data_path + '/MetaData_normal.npy', allow_pickle=True)

    test_df = pd.read_csv(os.path.join('../Data/'+data_path, 'test.csv'), index_col=None, usecols=None)
    max_user_n = test_df['userID'].max()
    test_df['itemID'] = test_df['itemID'] + max_user_n + 1
    test_segment_df = test_df[['userID', 'itemID']]
    val_list = np.array(test_segment_df)

    val_user_data = {}
    for (user_id, item_id) in val_list:
        if user_id not in val_user_data:
            val_user_data[user_id] = [item_id]
        else:
            val_user_data[user_id].append(item_id)

    val_user_ids = list(val_user_data.keys())
    val_data = []
    for user_id in val_user_ids:
        temp = [user_id]
        temp.extend(val_user_data[user_id])
        val_data.append(temp)

    test_data = val_data
    print('Data has been loaded.')
    ##########################################################################################################################################
    model = Net(num_user, num_item, train_edge, user_item_dict, weight_decay,
                        ce_weight, kd_weight, t_decay,
                        v_feat, a_feat, t_feat, meta_data,
                        aggr_mode, weight_mode, fusion_mode,
                        num_routing, dropout,
                        has_act, has_norm, has_entropy_loss, has_weight_loss,
                        is_word,
                        dim_E, dim_C,
                        pruning).cuda()
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    max_hk = 0.0
    max_hk_recall = 0.0
    num_decreases = 0 
    for epoch in range(num_epoch):
        # with autograd.detect_anomaly():  # 开启异常检测
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, writer)
        if torch.isnan(loss):
            with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan'.format(learning_rate, weight_decay))
            break
        torch.cuda.empty_cache()

        val_precision, val_recall, val_ndcg, val_hk, val_hk_recall = full_t(epoch, model, 'Train', valid_ids, writer)
        val_precision, val_recall, val_ndcg, val_hk, val_hk_recall = full_vt(epoch, model, val_data, 'Val', writer)
        test_precision, test_recall, test_ndcg, test_hk, test_hk_recall = full_vt(epoch, model, test_data, 'Test', writer)

        if test_ndcg > max_NDCG:
            max_precision = test_precision
            max_recall = test_recall
            max_NDCG = test_ndcg
            max_hk = test_hk
            max_hk_recall = test_hk_recall
            num_decreases = 0
        else:
            if num_decreases > 20:
                print('parameter settings are', args)
                print('dropout: {0} \t lr: {1} \t Weight_decay:{2} =====> '
                      'Precision:{3:.4f} \t Recall:{4:.4f} \t NDCG:{5:.4f}\t'
                      'HK@20:{6:.4f}\t Hk_Recall@20:{7:.4f}\r\n'.
                      format(dropout, learning_rate, weight_decay,
                             max_precision, max_recall, max_NDCG,
                             max_hk, max_hk_recall))

                break
            else:
                num_decreases += 1
