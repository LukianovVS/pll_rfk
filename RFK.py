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
n_acc     = 20;
alp_amp   = 0.1
alp_accel = 0.1;
pll_ph0   = 0.2;                                                                # начальная ошибка по фазе
pll_f0    = 8;                                                                  # начальная ошибка по частоте
pll_accel0= 0.1;                                                                # начальная ошибка по ускорению
pll_snr0  = 30;                                                                 # начальная оценка С/Ш (НЕ ошибка)
""" параметры реального сигнала """
T_end = 60;
T_accel = [15, 45]
F_accel = 1 / 20
A_accel = 3; 
SNR   = [50, 35, 40]
T_snr = [10, 30]
"""============================================= 
   Далее производятся расчёты по исходным данным
   ============================================="""
"""------- calc other params ------------"""
T_acc = n_acc * T0;
N_point = int(T_end / T_acc);
time = np.arange(0, N_point, 1) * T_acc
pll_snr0 = SNR[0] - pll_snr0
"""-----------------------------------------------"""
""" медель реального сигнала (входное воздействие)"""
""" SNR """
N_snr = [int(T / T_acc) for T in T_snr ]
snr = np.array([SNR[0]] * N_point)
snr[N_snr[0] : N_snr[1]] = SNR[1]
snr[N_snr[1] :         ] = SNR[2]
""" динамика """
N_accel = [int(T / T_acc) for T in T_accel ]
t_accel = np.arange(0, N_accel[1] - N_accel[0], 1) * T_acc


RVA_step = np.matrix( [ [1, T_acc, (T_acc**2)/2],
                        [0,     1,        T_acc] ], float);
rva = np.matrix( [ [float('nan')] * N_point] * 3, float)
rva[0, 0] = 0;
rva[1, 0] = 0;
rva[2, 0          : N_accel[0]] = 0;
rva[2, N_accel[0] : N_accel[1]] = A_accel * np.sin( 2 * math.pi * F_accel * t_accel);
rva[2, N_accel[1] :           ] = 0;


for k in range(1, N_point):    
    rva[0:2, k] = RVA_step * rva[:, k - 1]

del SNR, T_snr, A_accel, F_accel, T_accel, RVA_step, N_accel, t_accel, N_snr
"""-----------------------------------------------"""
"""--------------- фильтрация --------------------"""
F = np.matrix( [ [1, T_acc,          (T_acc**2) / 2, 0],
                 [0,     1,                   T_acc, 0],
                 [0,     0, (1 - alp_accel * T_acc), 0], 
                 [0,     0,                       0, 1] ], float );

G = np.matrix( [ [0,         0], 
                [0,         0],
                [0, alp_accel],
                [0, alp_amp  ]], float )

X = np.matrix( [[float('nan')] * N_point] * 4 )

X[0, 0] = pll_ph0;
X[1, 0] = pll_f0;
X[2, 0] = pll_accel0;
X[3, 0] = pll_snr0;




"""--------------------------------------------"""
"""--------------- графики --------------------"""
t2 = np.matrix(time)
plt.figure(1)
plt.clf()
plt.subplot(4,1,1)
plt.plot(time, rva[0,:].A1, label = 'input')
plt.plot(time,  X[0, :].A1, label = 'est')
plt.legend(), plt.grid(), plt.xlabel('t, sec'), plt.ylabel('ph, cycles')


plt.subplot(4,1,2)
plt.plot(time, rva[1,:].A1, label = 'input')
plt.plot(time,  X[1, :].A1, label = 'est')
plt.legend(), plt.grid(), plt.xlabel('t, sec'), plt.ylabel('fd, Hz')

plt.subplot(4,1,3)
plt.plot(time, rva[2,:].A1, label = 'input')
plt.plot(time,  X[2, :].A1, label = 'est')
plt.legend(), plt.grid(), plt.xlabel('t, sec'), plt.ylabel('accel, Hz/sec')

plt.subplot(4,1,4)
plt.plot(time,     snr   , label = 'input')
plt.plot(time, X[3, :].A1, label = 'est')
plt.legend(), plt.grid(), plt.xlabel('t, sec'), plt.ylabel('SNR, dBHz')



