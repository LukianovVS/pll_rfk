# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:07:19 2019

@author: lvs
"""
import numpy as np
import math
from math import pi as PI 

""" ------------------------------------------------------------------- 
    статистический эквивалент фазового дискриминатора
    rva - реальные фаза, частота, ускорение. Матрица [3, 1]
    snr - реальное SNR
    X   - оценка фазы, частоты, ускорения, SNR, на интервале измерений. 
          Матрица [3, 1]
   --------------------------------------------------------------------"""
def equivalent_ph(rva, snr, X, T):
    d_ph = float(rva[0] - X[0])
    d_w  = PI * float(rva[1] - X[1])
    d_ph_est = 2 * snr * T * math.sin(d_ph + d_w * T / 2) * np.sinc( (d_w * T / 2) / PI )
    return d_ph_est     