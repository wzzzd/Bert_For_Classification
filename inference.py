# -*- coding: UTF-8 -*-


from transformers import AdamW
from transformers import BertTokenizer
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup
import torch
from transformers import BertTokenizer, BertModel
import torch
from discriminator import Discriminator
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os 
from config import *

device = torch.device('cuda:1')


def main():

    # 读取输入数据
    df_src_test = pd.read_csv('./src/discriminator/data/performance_review/train.src', sep='\t').fillna('')
    src_test = df_src_test['src'].tolist()
    df_tgt_test = pd.read_csv('./src/discriminator/data/performance_review/train.tgt', sep='\t').fillna('')
    tgt_test = df_tgt_test['tgt'].tolist()

    print('read data...complete')
    src, tgt = data2batch(src_test, tgt_test, batch_size=16)
    print('data process...complete')
    print(len(src))
    print(len(tgt))
    # 读取序列分类的模型
    print('loading model...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # tokenizer.save_pretrained('./src/discriminator/checkpoints/tokenizer/')
    model = BertForSequenceClassification.from_pretrained('./src/discriminator/checkpoints/epoch_444/') #epoch_444

    # # 冻结base model的参数
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # 切换到gpu
    model.to(device)
    model.eval()

    # 遍历batch
    i = 0
    tgt_result = []
    tgt_prob = []
    print('inference start ...')
    for bs_src, bs_tgt in zip(src, tgt):

        # 输入
        encoding = tokenizer(bs_src, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)

        # 输出
        labels = torch.LongTensor(bs_tgt).to(device)
        outputs = model(input_ids, labels=labels)
        outputs = outputs[1]    #(batch size, label num)
        # print(outputs)
        # print(outputs)
        lab_outputs = torch.argmax(outputs,dim=1).cpu().numpy().tolist()
        lab_prob = torch.nn.functional.softmax(outputs, dim=1)
        lab_prob = lab_prob[:,0].cpu().detach().numpy().tolist()

        # print(lab_outputs)
        assert len(bs_src) == len(lab_outputs), "length is not equational between src and tgt in current batch."

        tgt_result.extend(lab_outputs)
        tgt_prob.extend(lab_prob)
        i += 1

    print('inference end ...')
    # score_current = get_valid_score(src, tgt, model, tokenizer)
    # print('test score:%s' %(score_current))

    # 写到文件
    print(len(src_test))
    print(len(tgt_result))
    assert len(src_test) == len(tgt_result), "length is not equational between src and pred."
    assert len(src_test) == len(tgt_test), "length is not equational between src and tgt."
    write_content = []
    for i in range(len(src_test)):
        tmp_src = src_test[i].replace('[SEP]','\t')
        tmp_tgt = str(tgt_test[i])
        tmp_pred = str(tgt_result[i])
        tmp_prob = str(tgt_prob[i])
        tmp_str = '\t'.join([tmp_src, tmp_tgt, tmp_pred, tmp_prob])
        write_content.append(tmp_str)
    write(write_content, './src/discriminator/data/pred/pred.csv', 'src_1\tsrc_2\ttgt\tpred\tprob')
    print('write data ...')



def get_valid_score(src, tgt, model, tokenizer):

    count = 0
    count_match = 0
    mix_dict = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
    for bs_src, bs_tgt in zip(src, tgt):

        # 输入
        encoding = tokenizer(bs_src, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        # groundtruth
        # bs_tgt_split = [x.split('[SEP]') for x in bs_tgt]
        # labels = [ [ int(y) for y in x] for x in bs_tgt_split]
        # 输出
        labels = torch.LongTensor(bs_tgt).to(device)
        outputs = model(input_ids, labels=labels)
        outputs = outputs[1]
        # print(outputs)
        # print(outputs.size())
        lab_outputs = torch.argmax(outputs,dim=1).cpu().numpy().tolist()
        # print(lab_outputs)
        # print(bs_tgt)
        
        # 计算准确率
        for x,y in zip(lab_outputs, bs_tgt):
            # print('pred:',x)
            # print('grou:',y)
            if x == y:
                count_match += 1

            if x==y and x==0:
                mix_dict['tp'] += 1
            if x==y and x==1:
                mix_dict['tn'] += 1
            if x!=y and x==0:
                mix_dict['fp'] += 1
            if x!=y and x==1:
                mix_dict['fn'] += 1

        count += len(bs_src)
    percision = mix_dict['tp']/(mix_dict['tp']+mix_dict['fp']) if (mix_dict['tp']+mix_dict['fp']) != 0 else 0
    recall = mix_dict['tp']/(mix_dict['tp']+mix_dict['fn']) if (mix_dict['tp']+mix_dict['fn']) != 0 else 0
    f1 = 2*percision*recall/(percision+recall) if (percision+recall) != 0 else 0
    print('percision:',percision)
    print('recall:',recall)
    print('f1:',f1)
    print(mix_dict)

    score = count_match/count
    return score






def write(text, path, col=''):
    with open(path, 'w', encoding='utf8') as f:
        if col != '':
            f.write(col+'\n')
        for line in text:
            f.write(line+'\n')



def data2batch(src, tgt, batch_size=16):

    assert len(src) == len(tgt), "length is not equation between src and tgt."

    # 获取索引
    s = 0
    e = 0
    last_num = 0
    d_group = []
    for i in range(len(src)):
        if i == 0 :
            continue
        if i % batch_size == 0:
            e = i
            s = last_num
            last_num = i
            d_group.append([s,e])
        if i == len(src)-1:
            d_group.append([last_num,i+1])
    # print(d_group[:100])

    # 切分数据
    src_groups = []
    tgt_groups = []
    for g in d_group:
        src_groups.append(src[g[0]:g[1]])
        tgt_groups.append(tgt[g[0]:g[1]])

    
    return src_groups, tgt_groups







if __name__ == '__main__':
    main()












