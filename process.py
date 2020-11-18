# -*- coding: UTF-8 -*-


import os 
import pandas as pd
import torch
import logging
from torch.nn import functional as F
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from config import Config
# from DataManager import DataManager
from Discriminator import Discriminator


Config = Config()
device = torch.device(Config.device)



def train(src_train, tgt_train, src_test, tgt_test, src_valid, tgt_valid):
    """
    训练过程
    """

    # 读取模型
    print('loading model...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer.save_pretrained(Config.path_tokenizer)
    model = Discriminator.from_pretrained('bert-base-chinese')
    model.save_pretrained(Config.path_bert)
    # 配置优化器
    optimizer = AdamW(model.parameters(), lr=Config.lr)
    # 切换到gpu
    model.to(device)
    # 启动训练模式
    model.train()

    # 冻结base model的参数
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    print('start training..')
    best_f1 = 0
    for epo in range(Config.epoch):
        i = 0
        for bs_src, bs_tgt in zip(src_train, tgt_train):
            # print(11)
            # print('current iter:', i)
            # 输入
            encoding = tokenizer(bs_src, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            labels = torch.LongTensor(bs_tgt).to(device)
            # print(encoding)

            # 定义loss，并训练
            optimizer.zero_grad()   # 梯度清零
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            # print(outputs)
            loss = outputs[1]
            loss.backward()
            optimizer.step()
            
            # 验证效果
            if i % 100 == 0:
                f1, acc = get_score(src_valid, tgt_valid, model, tokenizer)
                print('current epoch: %s/%s  iter:%s/%s  loss:%.6f  valid f1:%.3f  acc:%.3f' %(epo, Config.epoch, i, len(src_train), loss.item(), f1, acc))
                if f1 > best_f1:
                    save_path = os.path.join(Config.path_save_model, 'epoch_'+str(epo))
                    model.save_pretrained(save_path)
                    best_f1 = f1
                    print('save model success! ')
            # 测试效果
            if i % 500 == 0:
                f1, acc = get_score(src_test, tgt_test, model, tokenizer)
                print('current epoch: %s/%s  iter:%s/%s  loss:%.6f  test f1:%.3f  acc:%.3f' %(epo, Config.epoch, i, len(src_train), loss.item(), f1, acc))
            i += 1
        
        # 若本次迭代没有保存模型文件
        # save_path = os.path.join(Config.path_save_model, 'epoch_'+str(epo))
        # if not os.path.exists(save_path):
        #     model.save_pretrained(save_path)
    print('training end..')


def get_score(src, tgt, model, tokenizer):
    """
    计算在验证集上的准确率
    """
    # print('valid len:', len(srciqi))
    i = 0
    count = 0
    count_match = 0
    lab_groundtruth = []
    lab_predict = []
    for bs_src, bs_tgt in zip(src, tgt):
        # print('valid iter:',i)
        # 输入
        encoding = tokenizer(bs_src, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        labels = torch.LongTensor(bs_tgt).to(device)
        # 输出
        outputs = model(input_ids, labels=labels)
        outputs = outputs[0]
        lab_outputs = torch.argmax(outputs,dim=1).cpu().numpy().tolist()
        
        assert len(bs_tgt) == len(lab_outputs), "valid set: length difference between tgt and pred, in batch:%s" %str(i)
        lab_groundtruth.extend(bs_tgt)
        lab_predict.extend(lab_outputs)

        # 计算准确率
        for x,y in zip(lab_outputs, bs_tgt):
            if x == y:
                count_match += 1
        count += len(bs_src)

        i += 1

    # 计算acc
    acc = count_match/count
    # 计算f1
    f1 = f1_score(lab_groundtruth, lab_predict, average='macro')
    return f1, acc


def eval():
    pass


def infer():
    pass






if __name__ == '__main__':
    train()





