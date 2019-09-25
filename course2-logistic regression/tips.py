# !/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :19/9/20 上午12:33
# !@Author  :CHAMPLOO
# !@File    :tips.py


# 1. 断言

b = 0
assert (isinstance(b, float) or isinstance(b, int))  # #b的类型是float或者是int

# 2. 打印
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
