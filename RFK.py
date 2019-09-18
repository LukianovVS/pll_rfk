# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:35:06 2019

@author: Lukyanov VS
"""
"""---------- input lib-files ------------"""
import numpy as np
import matplotlib.pyplot as plt
import math
"""--------------- const -----------------"""
T0    = 1e-3;

"""-------- params of the model ---------"""
""" параметры ФАП """
n_acc = 20;
alp   = 0.1;
pll_ph0 = 0.2;
pll_f0  = 10;
pll_a0  = 0.1;
""" параметры реального сигнала """
T_end = 60;
T_accel = [15, 45]
F_accel = 1 / 20
A_accel = 3; 
SNR   = [50, 35, 40]
T_snr = [10, 30]
"""------- calc other params ------------"""
T_acc = n_acc * T0;
N_point = int(T_end / T_acc);


F = np.array( [ [1, T_acc,    (T_acc**2) / 2, 0],
                [0,     1,             T_acc, 0],
                [0,     0, (1 - alp * T_acc), 0], 
                [0,     0,                 0, 1] ], float );

G = np.array( [ [0] * 4 ] * 4, float )


time = np.arange(0, N_point, 1) * T_acc


""" медель реального сигнала (входное воздействие)"""
""" SNR """
N_snr = [int(T / T_acc) for T in T_snr ]
snr = np.array([SNR[0]] * N_point)
snr[N_snr[0] : N_snr[1]] = SNR[1]
snr[N_snr[1] :         ] = SNR[2]
""" динамика """
N_accel = [int(T / T_acc) for T in T_accel ]
t_accel = np.arange(0, N_accel[1] - N_accel[0], 1) * T_acc


RVA_step = np.array( [ [1, T_acc, (T_acc**2)/2],
                       [0,     1,        T_acc] ], float);
rva = np.array( [ [float('nan')] * N_point] * 3, float)
rva[0, 0] = 0;
rva[1, 0] = 0;
rva[2, 0          : N_accel[0]] = 0;
rva[2, N_accel[0] : N_accel[1]] = A_accel * np.sin( 2 * math.pi * F_accel * t_accel);
rva[2, N_accel[1] :           ] = 0;
rva0 = np.array( [[0]*1] * 3, float)

for k in range(1, N_point):    
    rva0[:, 0] = rva[:, k - 1]
    tmp = np.dot(RVA_step, rva0)
    rva[0:2, k] = tmp[0:2, 0]

del SNR, T_snr, A_accel, F_accel, T_accel, RVA_step, N_accel, t_accel, rva0
del tmp, N_snr

plt.figure(1)
plt.subplot(4,1,1)
plt.plot(time, rva[0,:])
plt.title('Входное воздействие')
plt.xlabel('t, sec')
plt.ylabel('ph, cycles')
plt.grid()
plt.subplot(4,1,2)
plt.plot(time, rva[1,:])
plt.xlabel('t, sec')
plt.ylabel('fd, Hz')
plt.grid()
plt.subplot(4,1,3)
plt.plot(time, rva[2,:])
plt.xlabel('t, sec')
plt.ylabel('accel, Hz/sec')
plt.grid()
plt.subplot(4,1,4)
plt.plot(time, snr)
plt.xlabel('t, sec')
plt.ylabel('SNR, dBHz')
plt.grid()





