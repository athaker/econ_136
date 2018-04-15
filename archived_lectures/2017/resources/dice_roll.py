#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 03:35:17 2017

@author: athaker
"""

import random as r
import numpy as np

import matplotlib.pyplot as plt
from numpy.random import randint, normal, uniform
t_max = 6
initial_value = 100.0
random_numbers = normal(size=t_max) * 0.07
multipliers = 1 + random_numbers
values = initial_value * np.cumprod(multipliers)
plt.xlabel('t')
values = np.insert(values,0,initial_value)
#ax = plt.plot(values)
"""
values[0]
values[1]
values[2]
values[3]
values[4]
values[5]
values[6]
"""
"""
ax = plt.plot(values[0:2])
ax = plt.plot(values[0:3])
ax = plt.plot(values[0:4])
ax = plt.plot(values[0:5])
ax = plt.plot(values[0:6])
"""

"""
# Log returns example
from numpy import log, exp, cumsum

t_max = 100
volatility = 1.0 / 100.0
initial_value = 100.0
r = normal(size=t_max) * volatility
y = initial_value * exp(cumsum(r))
plt.xlabel('t')
plt.ylabel('y')
ax = plt.plot(y)
"""