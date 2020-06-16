# -*- coding: utf-8 -*-
# author: mew

import numpy as np

drainage = 10  # mm/time_step

def model_test(rain_tot, rain_now, depth_pre, **kwargs):
    """ 本模型仅用于阈值划分，简单处理 """
    if 'verbose' in kwargs and kwargs['verbose']:
        print("-"*10 + "\ngenerated_flux_ratio:")
        print((rain_tot - rain_now + 50) / (rain_tot + 50))
        print("-"*10)

    return np.maximum(rain_now * (rain_tot - rain_now + 50) / (rain_tot + 50) + \
                      depth_pre - drainage, 0)

def model_test2(rain, depth_pre, **kwargs):
    """ 本模型仅用于简单处理 """
    res = np.zeros_like(rain)
    res[0] = model_test(rain[0], rain[0], depth_pre, **kwargs)
    for i in range(1, res.shape[0]):
        res[i] = model_test(rain[:(i+1)].sum(axis=0), rain[i], res[i-1], **kwargs)

    return res



def cal(rain_tot: np.ndarray, rain_now: np.ndarray, depth_pre: np.ndarray, **kwargs):
    """
    根据降雨量，推算积水深度
    例如：0时刻开始降雨，当前降雨时段为 (t0, t1], 其中 0 <= t0 < t1

    :param rain_tot: (lat, lon)
        连续降雨总量（包含当前时段的降雨量），即 （0, t1] 时刻的降雨量
    :param rain_now: (lat, lon)
        当前降雨时段的降雨量， 即 (t0, t1] 时刻的降雨量
    :param depth_pre: (lat, lon)
        当前降雨时段起始时刻的积水深度， 即 t0 时刻的积水深度
    :param kwargs:
    :return:
        当前降雨时段结束时刻的积水深度， 即 t1 时刻的积水深度

    >>> rain_tot = np.array([0, 10, 20, 30, 40, 50]).reshape(2, -1)
    >>> rain_now = np.array([0, 5, 20, 15, 0, 5]).reshape(2, -1)
    >>> depth_pre = np.array([0] * 4 + [10, 30]).reshape(2, -1)
    >>> cal(rain_tot, rain_now, depth_pre, verbose=True)
    ----------
    generated_flux_ratio:
    [[1.         0.91666667 0.71428571]
     [0.8125     1.         0.95      ]]
    ----------
    array([[ 0.        ,  0.        ,  4.28571429],
           [ 2.1875    ,  0.        , 24.75      ]])

    """
    assert (rain_tot.ndim == 2)
    assert (rain_tot.shape == rain_now.shape == depth_pre.shape)
    assert (np.min(rain_tot - rain_now) >= 0)
    assert (np.min(rain_now) >= 0)

    return model_test(rain_tot, rain_now, depth_pre, **kwargs)

def cal2(rain: np.ndarray, depth_pre: np.ndarray, **kwargs):
    """
    根据等间隔时间（1h）降雨量，推算每个时刻的积水深度

    :param rain:
    :param depth_pre:
    :param kwargs:
    :return:

    >>> rain = np.array([[[0, 10, 20], [30, 40, 50]], [[0, 5, 20], [15, 0, 5]],
    ... *[[[0]*3]*2]*5, ])
    >>> depth_pre = np.array([[0, 0, 0], [0, 10, 30]])
    >>> cal2(rain, depth_pre).shape
    (7, 2, 3)
    >>> cal2(rain, depth_pre)
    array([[[ 0,  0,  4],
            [ 8, 22, 45]],
    <BLANKLINE>
           [[ 0,  0,  9],
            [10, 12, 39]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  2, 29]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0, 19]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  9]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  0]]])

    """
    assert(rain.ndim==3)
    assert(rain.shape[1:] == depth_pre.shape)
    assert(np.min(rain) >= 0)

    return model_test2(rain, depth_pre, **kwargs)