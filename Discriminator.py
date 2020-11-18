# -*- coding: UTF-8 -*-

# from transformers import BertForSequenceClassification
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertPreTrainedModel
from config import Config
# from train import device


_TOKENIZER_FOR_DOC = "BertTokenizer"


class Discriminator(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        arg_config = Config()                                                                           # 读取配置文件
        self.device_num = torch.device(arg_config.device if torch.cuda.is_available() else 'cpu')       # 指定GPU or CPU
        self.batch_size = arg_config.batch_size                                                         # batch size
        
        # bert
        self.bert = BertModel(config)                                                                   # bert模型
        self.bert_hidden_size = arg_config.bert_hidden_size                                             # bert输出层大小
        self.dropout = nn.Dropout(arg_config.hidden_dropout_prob)                                       # dropout层
        # self.fc1 = nn.Linear(arg_config.bert_hidden_size, arg_config.fc1_size)
        # self.fc2 = nn.Linear(arg_config.fc1_size, arg_config.fc2_size)
        self.cls = nn.Linear(arg_config.bert_hidden_size, arg_config.num_labels)                        # 分类器
        self.num_labels = arg_config.num_labels                                                         # 类别数量

        # lstm layer
        self.use_lstm = arg_config.use_lstm                                                             # 是否使用lstm
        self.lstm_hidden_size = arg_config.lstm_hidden_size                                             # lstm隐藏层大小
        self.lstm_layer = arg_config.lstm_layer                                                         # lstm层数
        self.bi_num = arg_config.bi_num                                                                 # 双向lstm层数
        self.lstm = nn.LSTM(input_size=self.bert_hidden_size, 
                                hidden_size=self.lstm_hidden_size, 
                                num_layers=self.lstm_layer, 
                                bidirectional=arg_config.lstm_bidirectional)                            # LSTM层
        self.lstm_fc = nn.Linear(self.lstm_hidden_size * self.bi_num , self.lstm_hidden_size)           # 全连接层
        self.lstm_cls = nn.Linear(self.lstm_hidden_size, arg_config.num_labels)                         # 分类器
        
        # 初始化权重参数
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )                                                                                               #(batch_size, num_word, 768)


        # 获取最后一层作为bert的输出
        if self.use_lstm:
            sequence_output = outputs[0]                                                                #(batch_size, num_word, 768)
            # 填充batch size
            padding_batch_len = 0
            if sequence_output.shape[0] < self.batch_size:
                token_len = sequence_output.shape[1]
                padding_batch_len = self.batch_size - sequence_output.shape[0]
                zero_array = torch.zeros(padding_batch_len, token_len, sequence_output.shape[2]).to(self.device_num)
                sequence_output = torch.cat([sequence_output, zero_array], dim=0)

            # 加一层双向的LSTM
            h0 = torch.randn(self.lstm_layer * self.bi_num, self.batch_size, self.lstm_hidden_size, requires_grad = True).to(self.device_num)
            c0 = torch.randn(self.lstm_layer * self.bi_num, self.batch_size, self.lstm_hidden_size, requires_grad = True).to(self.device_num)

            # 添加lstm层
            sequence_output = sequence_output.transpose(0,1).contiguous().to(self.device_num)           #(num_word, batch_size, 768)
            sequence_output, hidden = self.lstm(sequence_output, (h0, c0))                              # (seq_num, batch_size, lstm_hidden_size)
            sequence_output = sequence_output.transpose(0,1).contiguous().to(self.device_num)           # (batch_size, seq_num, lstm_hidden_size)
            
            # 获取序列的最后一个cell
            sequence_output = sequence_output[:,-1,:].view(self.batch_size, -1)                         #(batch size, lstm_hidden_size)

            # 全连接层
            fc = self.lstm_fc(sequence_output)
            logits = self.lstm_cls(fc)                                                                  # (batch_size, num_labels)

        else:
            pooled_output = outputs[1]                                                                  # (batch_size, 768)
            pooled_output = self.dropout(pooled_output)
            # pooled_output = self.fc1(pooled_output)
            # pooled_output = self.fc2(pooled_output)
            logits = self.cls(pooled_output)                                                            # (batch_size, num_labels)

        # 计算损失
        loss = torch.zeros(1, ).to(self.device_num)
        if labels is not None:
            # 裁剪batch
            if self.use_lstm:
                logits = logits[:(self.batch_size-padding_batch_len),:]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (logits, loss)

















