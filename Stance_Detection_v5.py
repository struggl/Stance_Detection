# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report,f1_score
#添加库路径
import sys
sys.path.append('D://Stance_Detection_v5')
import time

import modeling
import hyper_params as hps
import utils

class Stance_Detection(object):
    def __init__(self):
        self.sess=tf.Session()
        self.model_path=hps.MODEL_PATH
        self.batch=0
        
    def reset_graph(self):
        tf.reset_default_graph()

    def get_log(self):
        self.writer.close()
    
    def build_dataset(self):
        #从excel文件中读取数据集并构建tf.data.Dataset
        self.train_data=pd.read_excel(hps.DATA_PATH+'/id_cut_train_data.xlsx')
        self.test_data=pd.read_excel(hps.DATA_PATH+'/id_cut_test_data.xlsx')
        fw_inputs,bw_inputs,lengths,stances=utils.make_batch(id_cut_data=self.train_data,
                                                             batch_size=len(model.train_data))
        self.Train_data=tf.data.Dataset.from_tensor_slices({
                'fw_inputs':fw_inputs,
                'bw_inputs':bw_inputs,
                'lengths':lengths,
                'stances':stances})
        fw_inputs,bw_inputs,lengths,stances=utils.get_inputs_for_testing(self.test_data)
        self.Test_data=tf.data.Dataset.from_tensor_slices({
                'fw_inputs':fw_inputs,
                'bw_inputs':bw_inputs,
                'lengths':lengths,
                'stances':stances})
    
    def save(self,path=hps.MODEL_PATH):
        #存储模型
        self.saver=tf.train.Saver()
        self.saver.save(self.sess,path)
        
    def load(self,path=hps.MODEL_PATH):
        #读取模型
        self.saver=tf.train.Saver()
        self.saver.restore(self.sess,path)
        
    def add_placeholder(self):
        #添加模型占位符
        self.fw_inputs=tf.placeholder(tf.int32,[hps.BATCH_SIZE,hps.MAX_LEN])
        self.bw_inputs=tf.placeholder(tf.int32,[hps.BATCH_SIZE,hps.MAX_LEN])
        self.lengths=tf.placeholder(tf.int32,[hps.BATCH_SIZE])
        self.stances=tf.placeholder(tf.int32,[hps.BATCH_SIZE])
        self.keep_prob=tf.placeholder(tf.float32)
        
    def evaluate(self):
        #对测试集进行预测
        st=time.time()
        predicts=[]
        test_data=self.Test_data.batch(hps.BATCH_SIZE)
        iterator=test_data.make_one_shot_iterator()
        one_batch=iterator.get_next()
        nbatch=len(self.test_data)//hps.BATCH_SIZE
        stances=[]
        for i in range(nbatch):  
            batch_data=self.sess.run(one_batch)     
            pred=self.sess.run(self.pred,feed_dict={self.fw_inputs:batch_data['fw_inputs'],
                                                    self.bw_inputs:batch_data['bw_inputs'],
                                                    self.lengths:batch_data['lengths'],
                                                    self.stances:batch_data['stances'],
                                                    self.keep_prob:1})
            stances.extend(list(batch_data['stances']))
            predicts.extend(list(pred))
            tf.logging.info('batch_{}'.format(i))
        end=time.time()
        tf.logging.info('\n\nUsed time:%.3f seconds' % (end-st))
        return (stances,predicts)
    
    def build_graph(self):
        #构建计算图的接口
        self.add_placeholder()
        self.build_dataset()
        self._graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(hps.LOG_PATH,tf.get_default_graph())
        
    def train(self,nepoch=1):
        '''训练模型的接口
        Args:
            nepoch:python int.数据集遍历次数
        '''
        st=time.time()
        train_data=self.Train_data.shuffle(1000).batch(hps.BATCH_SIZE).repeat(nepoch)
        iterator=train_data.make_one_shot_iterator()
        one_batch=iterator.get_next()
        i=0
        try:
            while True:
                batch_data=self.sess.run(one_batch)
                loss,summary,_=model.sess.run([self.loss,self.merged,self.train_op],
                                      feed_dict={self.fw_inputs:batch_data['fw_inputs'],
                                                 self.bw_inputs:batch_data['bw_inputs'],
                                                 self.lengths:batch_data['lengths'],
                                                 self.stances:batch_data['stances'],
                                                 self.keep_prob:hps.KEEP_PROB})
                self.writer.add_summary(summary, self.batch)
                self.batch+=1
                i+=1
                tf.logging.info('batch_%d_Loss:%.5f'%(i,loss))
        except tf.errors.OutOfRangeError:        
            end=time.time()
            tf.logging.info('\n\nUsed time of {} epochs:{} seconds'.format(nepoch,(end-st)))
            
    def train_batch(self,nbatch,batch_size=hps.BATCH_SIZE):
        '''指定batch数微调模型的接口
        Args:
            nbatch:python int.指定训练多少个batch
            batch_size:python int.指定batch_size
        '''
        train_data=self.Train_data.shuffle(1000).batch(batch_size)
        iterator=train_data.make_one_shot_iterator()
        one_batch=iterator.get_next()        
        i=0
        batch_data=self.sess.run(one_batch)
        loss,_=model.sess.run([self.loss,self.train_op],
                              feed_dict={self.fw_inputs:batch_data['fw_inputs'],
                                         self.bw_inputs:batch_data['bw_inputs'],
                                         self.lengths:batch_data['lengths'],
                                         self.stances:batch_data['stances'],
                                         self.keep_prob:hps.KEEP_PROB})
        tf.logging.info('batch_%d_Loss:%.5f'%(i,loss))
                
    def _graph(self):
        '''定义计算图的各个层
        '''
        with tf.variable_scope('L1/embedding_layer'):
            self.embedding=tf.get_variable('embedding',[hps.VOCAB_SIZE,hps.EMBED_SIZE])
            fw_inputs=tf.nn.embedding_lookup(self.embedding,self.fw_inputs)
            bw_inputs=tf.nn.embedding_lookup(self.embedding,self.bw_inputs)
            #fw_inputs=tf.nn.dropout(fw_inputs,self.embed_keep_prob)
            #bw_inputs=tf.nn.dropout(bw_inputs,self.embed_keep_prob)
        with tf.variable_scope('L2/BiLSTM_layer'):
            with tf.variable_scope('L2_1/fw_LSTM'):
                self.fw_cell=tf.nn.rnn_cell.BasicLSTMCell(hps.HIDDEN_SIZE)
                fw_h,fw_c=tf.nn.dynamic_rnn(self.fw_cell,fw_inputs,self.lengths,dtype=tf.float32)
            with tf.variable_scope('L2_2/bw_LSTM'):
                self.bw_cell=tf.nn.rnn_cell.BasicLSTMCell(hps.HIDDEN_SIZE)
                bw_h,bw_c=tf.nn.dynamic_rnn(self.bw_cell,bw_inputs,self.lengths,dtype=tf.float32)
            self.concat_h=tf.concat([fw_h,bw_h],-1)
          
        C_out=16
        self.conv_output=modeling.conv1d(inputs=self.concat_h,
                                         strides=(3,),
                                         C_outs=(C_out,),
                                         C_in=1,
                                         padding='VALID',
                                         scope='L3/conv1d_block_1',
                                         reuse=None)
        self.conv_output=tf.nn.dropout(self.conv_output,self.keep_prob)
        self.conv_output = modeling.normalize(self.conv_output)
            
        with tf.variable_scope('L4/Softmax'):
            self.weights=tf.get_variable('weights',[C_out,3])
            tf.summary.histogram('self.weigths',self.weights)
            self.bias=tf.get_variable('bias',[3])
            tf.summary.histogram('self.bias',self.bias)
            self.logits=tf.matmul(self.conv_output,self.weights)+self.bias
            tf.summary.histogram('self.logits',self.logits)
            self.pred=tf.argmax(self.logits,axis=1)
            self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.stances,logits=self.logits)
            self.loss=tf.reduce_sum(self.loss)
            tf.summary.scalar('self.loss',self.loss)
            
        trainable_vars=tf.trainable_variables()
        grads=tf.gradients(self.loss/tf.to_float(hps.BATCH_SIZE),trainable_vars)
        grads,_=tf.clip_by_global_norm(grads,hps.MAX_GRAD_NORM)
        optimizer=tf.train.RMSPropOptimizer(hps.LR)
        #optimizer=tf.train.AdadeltaOptimizer(hps.LR)
        self.train_op=optimizer.apply_gradients(zip(grads,trainable_vars))
        self.merged = tf.summary.merge_all()
        

if __name__=='__main__':  
    #构建计算图
    model=Stance_Detection()
    model.build_graph()
    #加载预训练好的模型并预测
    model.load()
    '''模型训练与存储
    model.train(nepoch=1)
    model.save()
    '''
    #对测试集进行预测
    stances,predicts=model.evaluate()
    #输出混淆矩阵
    print(classification_report(stances,predicts))
    #输出任务评价指标，即1类和2类的平均f1
    print(sum(f1_score(stances,predicts,average=None)[1:])/2)


   
    
        
        