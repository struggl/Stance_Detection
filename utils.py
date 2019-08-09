# -*- coding: utf-8 -*-
import collections
import jieba
import numpy as np
import pandas as pd
import random
import sys
sys.path.append('D://Stance_Detection_v5')

import hyper_params as hps

def calcu_classes(dataset,column_name):
	#查看划分的数据集话题与标签分布
    tg_set=set(dataset[column_name])
    tg_num=dict()
    for tg in tg_set:
        tg_num[tg]=len(np.where(dataset[column_name]==tg)[0])
    return tg_num


#读取停用词表
stopword_path=hps.DATA_PATH+'/stopword.txt'
stopwords=[line.strip() for line in open(stopword_path,'r',encoding='utf-8').readlines()]

def cut(_data):
	#jieba分词
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


def get_word_counter_and_dict(dataset,least_freq=2):
    '''
    dataset:pd.DataFrame,需包含['TEXT','TARGET']两列
    NOTE:the id 0 is left to padding and the id of char '<UNKNOWN>' is 1
    '''
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
        if counter[key]<=least_freq:
            del counter[key]
    words=['<PADDING>','<UNKNOWN>']
    words.extend(list(counter.keys()))
    word2id_dict=dict(zip(words,range(len(words))))
    id2word_dict=dict(zip(range(len(words)),words))
    return counter,word2id_dict,id2word_dict

def replace_word2id(_dataset,dataset_word2id_dict):
	#把原始文本的词替换成对应的编号
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
       

def clip_or_pad(input_i,max_len=hps.MAX_LEN):
    '''辅助函数：用于剪辑或者填充单个样本
    Args:
        inputs:(none)
        max_len:int 
    Return:
        res:(MAX_LEN)
    '''
    length=len(input_i)
    if len(input_i)>=max_len:
        length=max_len
        res=input_i[:max_len]
    else:
        res=np.array([0]*(max_len))
        res[:len(input_i)]=input_i
    return res,length   

id_cut_train_data=pd.read_excel(hps.DATA_PATH+'/id_cut_train_data.xlsx')
def get_max_len(id_cut_data):
	#统计每条样本的最大、最小、平均长度，采取平均长度到最大长度的均值作为clip_or_pad函数的max_len参数
    len_data=len(id_cut_data)
    id_cut_data.index=range(len_data)
    lengths=[]
    for i in range(len_data):
        sample=id_cut_train_data.ix[i]['TEXT']+id_cut_train_data.ix[i]['TARGET']
        id_sample=[int(w) for w in sample.strip().split()]
        lengths.append(len(id_sample))
    return lengths 

def make_batch(id_cut_data,batch_size=hps.BATCH_SIZE):
    '''make_batch for training,需要使用辅助函数clip_or_pad
    Args:
        id_cut_data:经过分词和编号化的样本,pd.DataFrame，必须有['TEXT'],['TARGET'],['STANCE']等列
        batch_size:int
    Return:
        fw_inputs:(batch_size,MAX_LEN)
        bw_inputs:(batch_size,MAX_LEN)
        lengths:(batch_size)
        stances:(batch_size)
    '''
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
    lengths=[]
    for i in range(len(targets)):
        fw_input=clip_or_pad(np.concatenate([targets[i],texts[i]]))
        bw_input=clip_or_pad(np.concatenate([targets[i][::-1],texts[i][::-1]]))
        fw_inputs.append(fw_input[0])
        bw_inputs.append(bw_input[0])
        lengths.append(fw_input[1])
    fw_inputs=np.array(fw_inputs)
    bw_inputs=np.array(bw_inputs)
    lengths=np.array(lengths)
    return fw_inputs,bw_inputs,lengths,stances

def _get_input_for_testing(id_cut_data_one_row):
    '''get input for testing,需要使用辅助函数clip_or_pad
    Args:
        id_cut_data_one_row:经过分词和编号化的一个样本,pd.DataFrame，必须有['TEXT'],['TARGET'],['STANCE']等列
    Return:
        fw_input:(MAX_LEN)
        bw_input:(MAX_LEN)
        length:(1)
        stance:(1)
    '''
    text=np.array([int(w) for w in id_cut_data_one_row['TEXT'].strip().split()])
    target=np.array([int(w) for w in id_cut_data_one_row['TARGET'].strip().split()])
    stance=id_cut_data_one_row['STANCE'] 
    fw_input=clip_or_pad(np.concatenate([target,text]))[0]
    bw_input,length=clip_or_pad(np.concatenate([target[::-1],text[::-1]]))
    return fw_input,bw_input,length,stance

def get_inputs_for_testing(id_cut_data):
    '''
    Args:
        id_cut_test_data:pd.DataFrame，一个batch的经过分词和编号化的测试数据集,
            必须有['TEXT'],['TARGET'],['STANCE']等列
    Return:
        fw_inputs:(batch_size,MAX_LEN)
        bw_inputs:(batch_size,MAX_LEN)
        lengths:(batch_size)
        stances:(batch_size)
    '''
    len_data=len(id_cut_data)
    range_len_data=range(len_data)
    id_cut_data.index=range_len_data
    fw_inputs=[]
    bw_inputs=[]
    lengths=[]
    stances=[]
    for i in range_len_data:
        res=_get_input_for_testing(id_cut_data.ix[i])
        fw_inputs.append(res[0])
        bw_inputs.append(res[1])
        lengths.append(res[2])
        stances.append(res[3])
    return np.array(fw_inputs),np.array(bw_inputs),np.array(lengths),np.array(stances)  

def get_test_data_by_topic(id_trg2real_trg,id_cut_test_data_path):
    '''
    Arg:
        id_trg2real_trg:dict.形如
            {'112 102 113 58 ': '俄罗斯叙利亚反恐行动',
             '165 ': 'IphoneSE',
             '4 5 ': '开放二胎',
             '46 36 ': '春节放鞭炮',
             '9 10 11 ': '深圳禁摩限电'}
        id_cut_test_data_path:pd.DataFrame.index必须为range(样本数).测试集数据。
            必须有['TARGET','TEXT','STANCE']三列.
            ['TARGET','TEXT']都已经过分词，词以空格隔开，'TARGET'形如：112 102 113 58 	
            ['TEXT']形如：2958 1 112 1 
    Return:
        dict.每个键都是一个话题，值是一个pd.DataFrame，存储着id_cut_test_data对应话题的所有行
        
    '''
    id_cut_test_data=pd.read_excel(id_cut_test_data_path)
    TARGETs=id_cut_test_data['TARGET']
    topic_index={}
    targets=list(set(TARGETs))#['4 5 ', '165 ', '9 10 11 ', '46 36 ', '112 102 113 58 ']
    for i in range(len(id_cut_test_data)):
        curtrg=id_cut_test_data.ix[i]['TARGET']
        for trg in targets:
            if trg == curtrg:
                if topic_index.get(trg) is None:
                    topic_index[trg]=[i]
                else:
                    topic_index[trg].append(i)
    keys=topic_index.keys()
    for key in keys:
        topic_index[key] = np.asarray(topic_index[key])
    test_data_by_topic={}
    for key in keys:
        test_data_by_topic[id_trg2real_trg[key]]=id_cut_test_data.ix[topic_index[key]]
    for key in test_data_by_topic.keys():
        test_data_by_topic[key].index=range(len(test_data_by_topic[key]))
    return test_data_by_topic
        
		
if __name__ == '__main__':
	#切分训练/测试集，并存储
	if hps.IF_SPLIT_DATASET!=1:
		data_path=hps.DATA_PATH+'/data.xlsx'
		data=pd.read_excel(data_path)
		len_data=np.arange(len(data))
		random.shuffle(len_data)#打乱len_data的顺序
		test_data=data.ix[len_data[:986]]
		train_data=data.ix[len_data[986:]]
		train_data.index=range(len(train_data))
		test_data.index=range(len(test_data))
		test_data.to_excel(hps.DATA_PATH+'/test_data.xlsx')
		train_data.to_excel(hps.DATA_PATH+'/train_data.xlsx')

	#对训练/测试集切词并存储
	if hps.BOOL_CUT!=1:
		train_data=pd.read_excel(hps.DATA_PATH+'/train_data.xlsx')
		test_data=pd.read_excel(hps.DATA_PATH+'/test_data.xlsx')
		cut_train_data=cut(_data=train_data)     
		cut_test_data=cut(_data=test_data) 
		cut_train_data.to_excel(hps.DATA_PATH+'/cut_train_data.xlsx')
		cut_test_data.to_excel(hps.DATA_PATH+'/cut_test_data.xlsx')

	#文本编号化并存储
	if hps.REPLACE_DATA!=1:
		cut_train_data=pd.read_excel(hps.DATA_PATH+'/cut_train_data.xlsx')
		cut_test_data=pd.read_excel(hps.DATA_PATH+'/cut_test_data.xlsx')
		train_word_counter,train_word2id_dict,train_id2word=get_word_counter_and_dict(cut_train_data)
		id_cut_train_data=replace_word2id(_dataset=cut_train_data,dataset_word2id_dict=train_word2id_dict)
		id_cut_test_data=replace_word2id(_dataset=cut_test_data,dataset_word2id_dict=train_word2id_dict)
		id_cut_train_data.to_excel(hps.DATA_PATH+'/id_cut_train_data.xlsx')
		id_cut_test_data.to_excel(hps.DATA_PATH+'/id_cut_test_data.xlsx')
    


         
        
    
    
    
    
    




