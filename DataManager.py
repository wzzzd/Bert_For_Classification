# -*- coding: UTF-8 -*-

import os
import re
import logging
import pandas as pd
from config import *



class DataManager(object):


    def __init__(self):

        # 读取训练数据
        self.config = Config()
        self.path = self.config.path_dataset
        self.batch_size = self.config.batch_size
        self.read_dataset()
        print('trian size len: %s'%str(len(self.src_train)))
        print('vaild size len: %s'%str(len(self.src_valid)))
        print('test size len: %s'%str(len(self.src_test)))


    def read_dataset(self):
        """
        读取数据集
        """
        path_train_src = os.path.join(self.path, 'train.src.csv')
        path_train_tgt = os.path.join(self.path, 'train.tgt.csv')
        path_valid_src = os.path.join(self.path, 'valid.src.csv')
        path_valid_tgt = os.path.join(self.path, 'valid.tgt.csv')
        path_test_src = os.path.join(self.path, 'test.src.csv')
        path_test_tgt = os.path.join(self.path, 'test.tgt.csv')
        self.src_train = self.read(path_train_src)
        self.tgt_train = self.read(path_train_tgt)
        self.src_valid = self.read(path_valid_src)
        self.tgt_valid = self.read(path_valid_tgt)
        self.src_test = self.read(path_test_src)
        self.tgt_test = self.read(path_test_tgt)


    def get_batch_data(self, c='train'):
        """
        获取按照 batch 数据
        """

        if c=='train':
            src = self.src_train
            tgt = self.tgt_train
        elif c=='valid':
            src = self.src_valid
            tgt = self.tgt_valid
        else:
            src = self.src_test
            tgt = self.tgt_test

        src_groups, tgt_groups = self.data2batch(src, tgt)
        # src_iter = iter(src_groups)
        # tgt_iter = iter(tgt_groups)

        return src_groups, tgt_groups


    def data2batch(self, src, tgt):
        """
        获取按照 batch 切分的数据
        """
        assert len(src) == len(tgt), "length is not equation between src and tgt."
        # 获取索引
        s = 0
        e = 0
        last_num = 0
        d_group = []
        for i in range(len(src)):
            if i == 0 :
                continue
            if i % self.batch_size == 0:
                e = i
                s = last_num
                last_num = i
                d_group.append([s,e])
            if i == len(src)-1:
                d_group.append([last_num,i+1])
        # 切分数据
        src_groups = []
        tgt_groups = []
        for g in d_group:
            tmp_src = [x[:100] for x in src[g[0]:g[1]]]
            # tmp_src = src[g[0]:g[1]]
            tmp_tgt = [ int(x) for x in tgt[g[0]:g[1]]]
            src_groups.append(tmp_src)
            tgt_groups.append(tmp_tgt)
        return src_groups, tgt_groups
        

    def read(self, path):
        """读取csv文件"""
        r = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                r.append(line.strip())
        return r