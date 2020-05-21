# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:31:54 2020

@author: yinjw
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import os
import pandas_profiling
import tensorflow as tf
import random
import math


os.chdir('D:\work\Kaggle\Beginner\HousePrice')
data=pd.read_csv('train.csv')


#View the whole dataset
# profile=pandas_profiling.ProfileReport(data)
# profile.to_file("D:\work\Kaggle\Beginner\HousePrice\output_file.html")

#delete the parameter with too many na，删掉缺失值太多的变量
data=data.drop(['Alley','Fence','FireplaceQu','MiscFeature','PoolQC'], axis=1)

#handle skewed parameter by log，取对数近似为正态分布
for i in range(0,len(data)):
    data.loc[i,'MiscVal']=math.log(data.loc[i,'MiscVal']+1)

#handle dummy variable，类别变量处理
data=pd.get_dummies(data,dummy_na=False,drop_first=True)


#handle na,用均值替代缺失值
for i in range(0,data.shape[1]):
    data.iloc[:,i].fillna(np.nanmean(data.iloc[:,i]),inplace=True)

#去掉缺失值太多的和0太多的，去掉关联度太高的
data=data.drop(['3SsnPorch','BsmtFinSF2','EnclosedPorch','Exterior2nd_CBlock','Exterior2nd_CmentBd','Exterior2nd_MetalSd','Exterior2nd_VinylSd','LowQualFinSF','PoolArea','SaleType_New','ScreenPorch'], axis=1)

#standerdize，把参数数值映射到[0,1]
for i in range(0,data.shape[1]):
    imax=data.iloc[:,i].max()
    imin=data.iloc[:,i].min()
    for j in range(0,len(data)):
        data.iloc[j,i]=(data.iloc[j,i]-imin)/(imax-imin)

#convert double to float
data=pd.DataFrame(data,dtype=np.float32)

#
# profile=pandas_profiling.ProfileReport(data)
# profile.to_file("D:\work\Kaggle\Beginner\HousePrice\output_file_final.html")

#sample，抽样，分出训练集和测试集。验证集本例暂不做了。。。
list1=np.arange(0,len(data),1)
list1=list(list1)
list2=random.sample(list1,1300)


train_data=data.loc[np.array(list2)]
train_x=train_data.sort_values('Id').drop(['Id','SalePrice'], axis=1)
train_y=train_data.sort_values('Id')['SalePrice']
test_data=data.loc[np.setdiff1d(list1,list2)]
test_x=test_data.sort_values('Id').drop(['Id','SalePrice'], axis=1)
test_y=test_data.sort_values('Id')['SalePrice']


# Parameters.学习率，训练轮数，显示的轮数。学习率大会报错？
learning_rate = 0.03
training_steps = 50000
display_step = 1000


# Weight and Bias, initialized randomly.初始化参数值
rng = np.random

W = tf.Variable(tf.ones([train_x.shape[1],1]), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Linear regression (Wx + b).
def linear_regression(x):
    return tf.matmul(tf.convert_to_tensor(np.array((x))),W) + b

# Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Stochastic Gradient Descent Optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(train_x.values)
        loss = mean_square(pred, train_y.values)
    
    # Compute gradients.
    gradients = g.gradient(loss, [W, b])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))


# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(train_x.values)
        loss = mean_square(pred, train_y.values)
        acc_mse = mean_square(tf.matmul(tf.convert_to_tensor(np.array((test_x.values))),W) + b, test_y.values)
        print("step: %i, loss: %f,  b: %f, acc_mse: %f" % (step, loss, b.numpy(),acc_mse.numpy()))

import matplotlib.pyplot as plt
# Graphic display
pred_test_y=tf.matmul(tf.convert_to_tensor(np.array((test_x.values))),W) + b
plt.plot(np.arange(0,len(test_y)), test_y, 'ro', label='Original data')
plt.plot(np.arange(0,len(test_y)), pred_test_y, label='Fitted data')
plt.legend()
plt.show()
