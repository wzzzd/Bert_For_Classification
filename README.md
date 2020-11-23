# Bert_For_Classification

本框架是基于Bert的文本分类模型框架。



# 依赖

    python3.6
	pandas==0.25.0
    torch==1.3.0
    transformers==3.0.2
    scikit-learn==0.23.2
可通过以下命令安装依赖包

    pip install -r requirements.txt


# 数据集
## Demo数据
* 数据集来自于项目 [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)，感谢开源。
* 是关于新闻标题的分类数据集，文本长度在20-30之间，共10个类别，每类2w条。
* 类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。
* 数据详情，请见上述开源项目。

## 自己的数据集
* 可使用本项目的处理方式，将数据集切分为3部分：train/valid/test
* 其中srt和tgt分开不同文件，并存储到 ./dataset/sample 目录
* 其中groundtruth的中文类别，需要手动做类别映射，映射成由0开始的int类型数字，示例可参看train.src.csv/train.tgt.csv/class.txt


# 模型
## 主要模型
本项目主要是封装了Bert并，将其整合成一个通用的文本序列分类的模块。具体模型，目前包含以下两个部分
* Bert + Classify，基于bert的分类器
* Bert + BiLSTM + Classsify，基于bert+bilstm的分类器

模型结构可以自行修改 ./model.py文件。
模型结构参数可以从 ./config.py中修改。


## 依赖模块
开源项目[transformers](https://github.com/huggingface/transformers)模块封装了Bert及其下游任务的各种API接口，如
|模块|说明|
|------|------|
| BertModel | Bert模型 |
| BertForSequenceClassification | 序列分类任务，如文本分类/回归 |
| BertForTokenClassification | 序列标注任务，如 |
| 序列标注任务，如NERBertForQuestionAnswering | 阅读理解任务，如SQuAD |
此处只列举了部分下游任务，详情可参考[官方文档](https://huggingface.co/transformers/model_doc/bert.html)



# Get Started
## 训练
准备好训练数据后，终端可运行命令

    python3 run.py

## 评估
加载已训练好的模型，并使用valid set作模型测试，输出文件到 ./dataset/eval/ 目录下。终端可运行命令

    python3 run.py --mode eval

## 预测
预测未知标签数据集，并保存为文件 ./dataset/infer/test.csv. 终端可运行命令

    python3 run.py --mode infer




