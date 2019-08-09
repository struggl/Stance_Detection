# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:37:55 2018

@author: 7758
"""
'''


'''
import numpy as np
import pandas as pd
import random

IF_SPLIT_DATASET=1

base_path='G://Stance_Detection/data'
if IF_SPLIT_DATASET!=1:
    data_path=base_path+'/data.xlsx'
    data=pd.read_excel(data_path)
    len_data=np.arange(len(data))
    random.shuffle(len_data)#打乱len_data的顺序
    test_data=data.ix[len_data[:986]]
    train_data=data.ix[len_data[986:]]
    train_data.index=range(len(train_data))
    test_data.index=range(len(test_data))
    test_data.to_excel(base_path+'/test_data.xlsx')
    train_data.to_excel(base_path+'/train_data.xlsx')


##查看划分的数据集话题与标签分布
train_data=pd.read_excel(base_path+'/train_data.xlsx')
test_data=pd.read_excel(base_path+'/test_data.xlsx')

def calcu_classes(dataset,column_name):
    tg_set=set(dataset[column_name])
    tg_num=dict()
    for tg in tg_set:
        tg_num[tg]=len(np.where(dataset[column_name]==tg)[0])
    return tg_num

calcu_classes(dataset=train_data,column_name='TARGET')
calcu_classes(dataset=test_data,column_name='TARGET')
calcu_classes(dataset=train_data,column_name='STANCE')
calcu_classes(dataset=test_data,column_name='STANCE')


'''
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += str(inside_code)
    return rstring
 
def p(s):
    s1 = strQ2B(s.decode())
    p = re.compile('[()]',re.S)
    s1 = p.sub('',s1)
    return s1
'''
#jieba分词
import jieba
stopword_path=base_path+'/stopword.txt'
stopwords=[line.strip() for line in open(stopword_path,'r',encoding='utf-8').readlines()]
def cut(_data):
    data=_data.copy()
    data.index=range(len(data))
    targets=[]
    texts=[]
    stances=[]
    for i in range(len(data)):
        target=[]
        for w in jieba.cut(data.ix[i]['TARGET']):
            if w not in stopwords:
                target.append(w)
        targets.append(' '.join(target))
        target=[]
        for w in jieba.cut(data.ix[i]['TEXT']):
            if w not in stopwords:
                target.append(w)
        texts.append(' '.join(target))
        if data.ix[i]['STANCE']=='FAVOR':
            stances.append(2)
        elif data.ix[i]['STANCE']=='AGAINST':
            stances.append(1)
        else:
            stances.append(0)
    return pd.DataFrame({'TARGET':targets,'TEXT':texts,'STANCE':stances})
BOOL_CUT=1
if BOOL_CUT!=1:
    cut_train_data=cut(_data=train_data)     
    cut_test_data=cut(_data=test_data) 
    cut_train_data.to_excel(base_path+'/cut_train_data.xlsx')
    cut_test_data.to_excel(base_path+'/cut_test_data.xlsx')

   
        

###
cut_train_data=pd.read_excel(base_path+'/cut_train_data.xlsx')
cut_test_data=pd.read_excel(base_path+'/cut_test_data.xlsx')
import collections
def get_word_counter_and_dict(dataset):
    counter=collections.Counter()
    len_dataset=len(dataset)
    dataset.index=range(len_dataset)
    for i in range(len_dataset):
        text=dataset.ix[i]['TEXT']
        target=dataset.ix[i]['TARGET']
        for w in text.strip().split():
            counter[w]+=1
        for w in target.strip().split():
            counter[w]+=1
    list_keys=list(counter.keys())
    for key in list_keys:
        if counter[key]<=2:
            del counter[key]
    words=['<UNKNOWN>']
    words.extend(list(counter.keys()))
    word2id_dict=dict(zip(words,range(len(words))))
    id2word_dict=dict(zip(range(len(words)),words))
    return counter,word2id_dict,id2word_dict
train_word_counter,train_word2id_dict,train_id2word=get_word_counter_and_dict(cut_train_data)  
test_word_counter,test_word2id_dict,test_id2word=get_word_counter_and_dict(cut_test_data)

###把原始文本的词替换成对应的编号
def replace_word2id(_dataset,dataset_word2id_dict):
    dataset=_dataset.copy()
    len_dataset=len(dataset)
    dataset.index=range(len_dataset)
    TARGETS=[]
    TEXTS=[]
    for i in range(len_dataset):
        target=dataset.ix[i]['TARGET'].strip().split()
        
        TARGET=''
        for w in target:
            if dataset_word2id_dict.get(w) is None:
                TARGET+=repr(dataset_word2id_dict['<UNKNOWN>'])+' '
            else:
                TARGET+=repr(dataset_word2id_dict[w])+' '
        TARGETS.append(TARGET)
        
        text=dataset.ix[i]['TEXT'].strip().split()
        
        TEXT=''
        for w in text:
            if dataset_word2id_dict.get(w) is None:
                TEXT+=repr(dataset_word2id_dict['<UNKNOWN>'])+' '
            else:
                TEXT+=repr(dataset_word2id_dict[w])+' '
        TEXTS.append(TEXT)
    return pd.DataFrame({'TARGET':TARGETS,'TEXT':TEXTS,'STANCE':dataset['STANCE']})

REPLACE_DATA=1
if REPLACE_DATA!=1:
    id_cut_train_data=replace_word2id(_dataset=cut_train_data,dataset_word2id_dict=train_word2id_dict)
    id_cut_test_data=replace_word2id(_dataset=cut_test_data,dataset_word2id_dict=test_word2id_dict)
    id_cut_train_data.to_excel(base_path+'/id_cut_train_data.xlsx')
    id_cut_test_data.to_excel(base_path+'/id_cut_test_data.xlsx')
        
    
##make_batch for training
id_cut_train_data=pd.read_excel(base_path+'/id_cut_train_data.xlsx')
id_cut_test_data=pd.read_excel(base_path+'/id_cut_test_data.xlsx')
def _make_batch(id_cut_data,batch_size=50):
    len_data=np.arange(len(id_cut_data))
    id_cut_data.index=range(len_data.shape[0])
    random.shuffle(len_data)
    batch_data=id_cut_data.ix[len_data[:batch_size]]
    batch_data.index=range(batch_size)
    texts=[]
    targets=[]
    stances=[]
    for i in range(batch_size):
        text=[int(w) for w in batch_data.ix[i]['TEXT'].strip().split()]
        target=[int(w) for w in batch_data.ix[i]['TARGET'].strip().split()]
        stance=batch_data.ix[i]['STANCE']
        texts.append(np.array(text))
        targets.append(np.array(target))
        stances.append(np.array(stance))
    targets=np.array(targets)
    texts=np.array(texts)
    stances=np.array(stances)
    fw_inputs=[]
    bw_inputs=[]
    for i in range(len(targets)):
        fw_inputs.append(np.concatenate([targets[i],texts[i]]))
        bw_inputs.append(np.concatenate([targets[i][::-1],texts[i][::-1]]))
    return np.array(fw_inputs),np.array(bw_inputs),stances

fw_inputs,bw_inputs,stances=_make_batch(id_cut_data=id_cut_train_data,batch_size=5)


##get input for testing
def _get_input_for_testing(id_cut_data_one_row):
    text=np.array([int(w) for w in id_cut_data_one_row['TEXT'].strip().split()])
    target=np.array([int(w) for w in id_cut_data_one_row['TARGET'].strip().split()])
    stance=np.array(id_cut_data_one_row['STANCE'])
    fw_inputs=[]
    bw_inputs=[]    
    fw_inputs.append(np.concatenate([target,text]))
    bw_inputs.append(np.concatenate([target[::-1],text[::-1]]))
    return np.array(fw_inputs),np.array(bw_inputs),stance

fw_input,bw_input,stance=_get_input_for_testing(id_cut_data_one_row=id_cut_train_data.ix[10])

##辅助函数：用于剪辑或者填充样本
def clip_or_pad(fw_inputs,bw_inputs,max_len):
    pass
    
#统计每条样本的最大、最小、平均长度，采取平均长度到最大长度的均值作为clip_or_pad函数的max_len参数
fw_inputs,bw_inputs,stances=_make_batch(id_cut_data=id_cut_train_data,batch_size=len(id_cut_train_data))
len_data=list(map(len,fw_inputs))
np.mean(len_data)#26.846
max(len_data)#77
min(len_data)#2   