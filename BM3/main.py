# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BM3', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu', nargs='?', default='0', help='gpu_id')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--ce_weight', type=float, default=0.2, help='Learning rate.')
    parser.add_argument('--kd_weight', type=float, default=0.2, help='Learning rate.')
    parser.add_argument('--t_decay', nargs='?', default=10, help='kd_weight.')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu,
        'learning_rate': args.l_r,
        'ce_weight': args.ce_weight,
        'kd_weight': args.kd_weight,
        't_decay': args.t_decay
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


