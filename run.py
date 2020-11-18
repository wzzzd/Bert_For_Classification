# -*- coding: UTF-8 -*-


import time
import numpy as np
import torch
import logging
from config import Config
from DataManager import DataManager
from process import train, eval, infer
import argparse

parser = argparse.ArgumentParser(description='Bert-based Classification')
# parser.add_argument('--model', default='TextCNN', type=str, required=False, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# args = parser.parse_args()


if __name__ == '__main__':

    # logging.basicConfig(filename='./sys.log', filemode='a', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  #日志配置
    Config = Config()

    # 设置随机种子保证结果
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    start_time = time.time()

    # 数据处理
    print('read data...')
    dm = DataManager()
    print('data process...')
    src_train, tgt_train = dm.get_batch_data('train')
    src_test, tgt_test = dm.get_batch_data('test')
    src_valid, tgt_valid = dm.get_batch_data('valid')
    print('trian batch size len: %s' %str(len(src_train)))
    print('vaild batch size len: %s' %str(len(src_valid)))
    print('test batch size len: %s' %str(len(src_test)))

    # 模式
    if Config.mode == 'train':
        train(src_train, tgt_train, src_test, tgt_test, src_valid, tgt_valid)
    elif Config.mode == 'eval':
        eval(src_test, tgt_test)
    elif Config.mode == 'infer':
        infer()
    else:
        print("have a good day ~")
