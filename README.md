# PyTorch Implementation of Feature Based NER with pretrained Bert

I know that you know [BERT](https://arxiv.org/abs/1810.04805).
In the great paper, the authors claim that the pretrained model does great on NER without fine-tuning.
It's even impressive, allowing for the fact that they don't use any autoregressive technique such as CRF.
We try to reproduce the result in a simple manner.

## Requirements
* python>=3.6 (Let's move on to python 3 if you still use python 2)
* pytorch==1.0
* pytorch_pretrained_bert==0.6.1
* numpy>=1.15.4

## Training & Evaluating

* STEP 1. Run the command below to download conll 2003 NER dataset.
```
bash download.sh
```
It should be extracted to `conll2003/` folder automatically.

* STEP 2. Run the command below to train and evaluate.
```
python train.py
```

## Results in the paper
<img src="bert_ner.png" align="left">

## Results

* You can check the classification outputs in [checkpoints](checkpoints).

|epoch|F1 score on conll2003 valid|
|--|--|
|1|0.2|
|2|0.75|
|3|0.84|
|4|0.88|
|5|0.89|
|6|0.90|
|7|0.90|
|8|0.91|
|9|0.91|
|10|0.92|
|11|0.92|
|12|0.93|
|13|0.93|
|14|0.93|
|15|0.93|
|16|0.92|
|17|0.93|
|18|0.93|
|19|0.93|
|20|0.93|
|21|**0.94**|
|22|**0.94**|
|23|0.93|
|24|0.93|
|25|0.93|
|26|0.93|
|27|0.93|
|28|0.93|
|29|**0.94**|
|30|0.93|
