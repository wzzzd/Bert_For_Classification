# model arg


class Config(object):

    def __init__(self):

        self.mode = 'train'                                            # 模式：train/eval/infer
        self.device = 'cuda:6'                                         # GPU卡号
        self.batch_size = 16                                           # batch size
        self.epoch = 50                                                # epoch数量
        self.lr = 1e-4                                                 # learning rate

        self.use_lstm = False                                          # 是否使用lstm
        self.lstm_hidden_size = 1024                                   # lstm隐藏层大小
        self.lstm_layer = 1                                            # lstm层数
        self.lstm_bidirectional = True                                 # 是否使用双向
        self.bi_num = 2                                                # 双向的lstm层数（默认不用修改）
        
        self.num_labels = 61                                           # 分类数
        self.hidden_dropout_prob = 0.1                                 # dropout比例
        self.bert_hidden_size = 768                                    # bert隐层大小
        self.fc1_size = 1024
        self.fc2_size = 512

        # transformer
        self.tf_heads = 12
        self.tf_layer_num = 6

        # 数据集
        self.path_dataset = './dataset/'                               # 数据集位置
        self.path_save_model = './checkpoint/model/'                   # 模型保存位置
        self.path_tokenizer = './checkpoint/tokenizer/'                # 分词器位置
        self.path_bert = './checkpoint/bert/'                          # 预训练的Bert-chinese-base参数





