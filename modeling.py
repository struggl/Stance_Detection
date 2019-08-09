# -*- coding: utf-8 -*-
import collections
import numpy as np
import re
import tensorflow as tf

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''定义batch_norm.Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def normal_embedding(inputs,
                     vocab_size,
                     embed_size,
                     zero_pad=True,
                     name='normal_embedding'):
	#提供填充零的embedding
    embedding=tf.get_variable(name,[vocab_size,embed_size])
    if zero_pad:
        embed = tf.concat((tf.zeros(shape=[1, embed_size]),
                           embedding[1:, :]), 0)
    embed=tf.nn.embedding_lookup(embedding,inputs)
    return embed
       	
def conv1d(inputs,
           strides=(3,),
           C_outs=(64,),
           C_in=1,
           padding='SAME',
           scope='conv1d_block',
           reuse=None):
    '''一维卷积接口
    Args:
        inputs:A 3D 'Tensor' with shape [batch_size,max_len,embed_size](alias:[N,T,C])
        strides:
    Returns:
        A Tensor with shape [inputs.get_shape().as_list[0],-1]
    '''
    inputs=tf.expand_dims(inputs,-1)#[N,T,C]==>[N,H,W,C=1]
    conv_res=[]
    N,H,W,C=inputs.get_shape().as_list()
    for i in range(len(strides)):
        h=strides[i]
        C_out=C_outs[i]
        with tf.variable_scope(scope+'_{}_with_stride_{}x{}'.format(i,strides[i],W),reuse=reuse):
            W_conv=tf.Variable(tf.truncated_normal([h,W,C_in,C_out],stddev=0.1))#[h,w,C_in,C_out]
            b_conv=tf.Variable(tf.truncated_normal([C_out],stddev=0.1))
            if padding=='VALID':
                h_conv=tf.nn.conv2d(inputs,
                                    W_conv,
                                    strides=[1,1,1,1],
                                    padding='VALID')#[N,(H-h+stride)/stride=(H-h+1),1,C_out]
                h_conv=tf.nn.relu(tf.nn.bias_add(h_conv,b_conv))#[N,H,1,C_out]
                #shape of ksize and strides in tf.nn.max_pool='NHWC'
                h_pool=tf.nn.max_pool(h_conv,
                                      ksize=[1,H-h+1,1,1],
                                      strides=[1,1,1,1],
                                      padding='VALID')#shape=[N,1,1,C_out]
                conv_res.append(h_pool)
            elif padding=='SAME':
                '''SAME模式下
                1.tf.nn.conv2d的strides参数必须设置为[1,1,W,1]，才能使得卷积以后的
                    shape=[N,H',1,C_out]
                2.tf.nn.max_pool的ksize和strides参数都必须设置为[1,H',1,1]才能使得
                    shape=[N,1,1,C_out]
                '''
                h_conv=tf.nn.conv2d(inputs,
                                    W_conv,
                                    strides=[1,1,W,1],
                                    padding='SAME')#[N,H',1,C_out]
                h_conv=tf.nn.relu(tf.nn.bias_add(h_conv,b_conv))#[N,H',1,C_out]
                #shape of ksize and strides in tf.nn.max_pool='NHWC'
                ksize_h,_=calcu_conv2ded_shape(input_sizes=(H,W),
                                               filter_sizes=(h,W),
                                               strides=(1,1),
                                               padding='SAME')#令ksize_h=H'
                h_pool=tf.nn.max_pool(h_conv,
                                      ksize=[1,ksize_h,1,1],
                                      strides=[1,ksize_h,1,1],
                                      padding='SAME')#shape=[N,1,1,C_out]
                conv_res.append(h_pool)                
    conv_res=tf.convert_to_tensor(conv_res)
    conv_res=tf.concat(conv_res,axis=-1)
    conv_res=tf.reshape(conv_res,[N,-1])
    return conv_res
 
def calcu_padding_size(input_size,
                       filter_size,
                       stride):
    '''calculate padding size when tf.nn.conv2d's padding=='SAME'
    Args:
        input_sizes:int 
        filter_sizes:int
        strides:int
    Returns:
        int.Height or Width after padding.
    '''
    output_size = (input_size+stride-1)//stride
    pad_need = max(0,(output_size - 1)*stride + filter_size - input_size)
    pad_left = pad_need//2
    pad_right = pad_need - pad_left
    return input_size+pad_left+pad_right

def calcu_conv2ded_shape(input_sizes,
                         filter_sizes,
                         strides,
                         padding='VALID'):
	#计算二维卷积后张量的shape
    H,W=input_sizes
    h_filter,w_filter=filter_sizes
    sh,sw=strides
    if padding=='VALID':
        H_=(H-h_filter+sh)//sh
        W_=(W-w_filter+sw)//sw
        return (H_,W_)
    elif padding=='SAME':
        H=calcu_padding_size(H,h_filter,sh)
        W=calcu_padding_size(W,w_filter,sw)
        H_=(H-h_filter+sh)//sh
        W_=(W-w_filter+sw)//sw
        return (H_,W_) 


    
                
                
                               
