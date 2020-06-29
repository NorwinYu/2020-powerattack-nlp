## 2020中国大学生保险数字科技挑战赛 - 科技赛道 - 随便取团队方案

Norwin 



## 赛题

**对话意图识别** 

https://powerattack.gobrand.top/

### 赛题数据

由于涉及到用户隐私及数据安全等问题，本次比赛不提供原始文本，而是使用字符编号及切词后的词语编号来表示文本信息。数据格式说明：

- id：通话流水号
- category：角色类型（0客户，1机器人）
- char：基于字的id
- word：基于词的id（其中词通过-进行连接）
- label：所属意图

训练数据和预测数据（测试集）都没有经过任何清洗。

### 最终排名

初赛总榜 Rank 14

初赛华东赛区第四

分数 0.29305968



## 线上最终方案

### 模型架构以及亮点

本次模型架构思路参考 chizhu 大佬在还在进行的腾讯广告算法大赛中的[分享](https://mp.weixin.qq.com/s/ISQjOGcc_spSNVeeg75d8w)。 我直接使用了根据这个架构复现的初版在腾讯赛初赛中的代码。总体思路为，词 id 根据 w2v sg 进行词向量训练，输出128维度词向量。对输入的句子，pad 100 后，喂入以 一层 Transformer （8头1024） + 一层bilstm (隐藏单元128) + Dense 的网络中。

不同于腾讯赛，此次比赛以 macro f1 为指标，并且类别多而数据不平衡。基于让神经网络更加鲁棒的思想，我并没有尝试对数据进行平衡，而是考虑选择合适的 loss，这是我们这个方案的亮点以及上分的关键。经过一些调试，我最终确定了一个 [focal loss](https://arxiv.org/pdf/1708.02002.pdf) 和 f1 loss 的混合 loss, 并且通过一定的加权使得两个loss在同一数量级，尽可能同步贡献梯度。使用这一loss相比于 ce loss，在不处理数据平衡情况下，线上单折从 0.2 增长到 0.27。在五折然后尝试不同折组合后，线上达到 0.29305968。其实根据实验， ce loss 和 f1 loss 的混合 loss已经能达到比较好的成绩，但是使用 focal loss 和 f1 loss 的混合 loss 能使得线上再提升几个千分点。[这个实验](https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb) 对于 使用f1 loss有一定启发意义。

```python
# focal loss code from https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
# f1 loss code from https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

def focal_loss(y_true, y_pred):
  gamma=2.
  alpha=.25  
  # Scale predictions so that the class probas of each sample sum to 1
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  
  # Clip the prediction value to prevent NaN's and Inf's
  epsilon = K.epsilon()
  y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
  # Calculate Cross Entropy
  cross_entropy = -y_true * K.log(y_pred)
  # Calculate Focal Loss
  loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
  # Compute mean loss in mini_batch
  return K.mean(loss, axis=1)

def focal_f1_loss(y_true, y_pred):
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())
  f1 = 2*p*r / (p+r+K.epsilon())
  alpha = 0.001
  return alpha *  (1 - K.mean(f1)) +  focal_loss(y_true, y_pred)
```

在我们的方案中，仅仅使用了word id 以及 单句信息。由于后续才真正投入比赛，没有进行上下文过多的尝试，这也是我们此次失利没有更进一步进入复赛的原因。



## 代码

### 环境

```
代码环境：python3, pandas, numpy, keras, sklearn, tensorflow, gensim
GPU环境：本次实验是在 Colab P100 上完成的，使用的最新的 tf 版本，单折运行时间大概一个小时不到。
```

### 流程

- `preprocess.py  `  数据提取，生成词向量
- `lstm_trf_focal_f1_loss.py` 训练数据准备（读取，pad，emb matrix等），模型构造，loss定义，K折训练
- `predict_lstm_trf.py` 每一折进行预测，权重搜索处理，保存每一折预测概率
- `predict_lstm_trf_5k.py`  k折概率平均，输出submission文件

### 代码参考

- 权重搜索处理 - [麻婆豆腐AI](https://mp.weixin.qq.com/s/mvFhLowwj7lqF762xz2KzQ)  感谢
- loss 定义 - [focal](https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py) [f1](https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric)
- Transformer 定义 [keras官方教程](https://keras.io/examples/nlp/text_classification_with_transformer/)
- [LN 定义](https://github.com/kpot/keras-transformer/blob/master/keras_transformer/transformer.py)



## 参考链接

https://powerattack.gobrand.top/

https://mp.weixin.qq.com/s/ISQjOGcc_spSNVeeg75d8w

https://mp.weixin.qq.com/s/mvFhLowwj7lqF762xz2KzQ

https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py

https://arxiv.org/pdf/1708.02002.pdf

https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb

https://github.com/kpot/keras-transformer/blob/master/keras_transformer/transformer.py

https://keras.io/examples/nlp/text_classification_with_transformer/