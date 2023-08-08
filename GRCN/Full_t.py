from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np

def full_t(epoch, model, prefix, valid_ids, writer=None):
    print(prefix+' start...')
    model.eval()

    with no_grad():
        precision, recall, ndcg_score, hk, hk_recall = model.accuracy(topk=20)
        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f} '
              'HK@20:{4:.4f} HK_Recall@20:{5:.4f} ---------------------------------'.format(
                epoch, precision, recall, ndcg_score, hk, hk_recall))
        if writer is not None:
            writer.add_scalar(prefix+'_Precition', precision, epoch)
            writer.add_scalar(prefix+'_Recall', recall, epoch)
            writer.add_scalar(prefix+'_NDCG', ndcg_score, epoch)
            writer.add_scalar(prefix+'_Hk', hk, epoch)
            writer.add_scalar(prefix+'_Hk_Recall', hk_recall, epoch)

            writer.add_histogram(prefix+'_visual_distribution', model.v_rep, epoch)
            writer.add_histogram(prefix+'_textual_distribution', model.t_rep, epoch)
            writer.add_histogram(prefix + 'representation_distribution', model.result, epoch)

            writer.add_embedding(model.result[valid_ids, :], global_step=epoch)

        return precision, recall, ndcg_score, hk, hk_recall



