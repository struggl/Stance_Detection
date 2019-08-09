# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 18:40:59 2018

@author: 7758
"""
#transforming raw data
IF_SPLIT_DATASET=1
BOOL_CUT=1
REPLACE_DATA=1
LOAD_FASTTEXT=1

#define paths
BASE_PATH='D://Stance_Detection_v5'
DATA_PATH=BASE_PATH+'/data'
MODEL_PATH=BASE_PATH+'/model'
LOG_PATH=BASE_PATH+'/log'

#define model hyperparameters
BATCH_SIZE=50#截至2018.12.01,batch_size=50得到最好效果
VOCAB_SIZE=5000
EMBED_SIZE=32
KEEP_PROB=0.85
MAX_LEN=50
HIDDEN_SIZE=64#LSTM层的隐藏向量维度
MAX_GRAD_NORM=5
LR=0.001

