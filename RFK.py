# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:35:06 2019

@author: Lukyanov VS
"""
"""---------- input lib-files ------------"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rmd 
import discriminator as discr 
"""--------------- const -----------------"""
T0    = 1e-3;
j_2pi = 1j * 2 * np.pi;
"""-------- params of the model ---------"""
""" параметры ФАП """
n_acc     = 2;
alp_amp   = 0.1
alp_accel = 0.1;
pll_ph0   = 0.1;                                                                # начальная ошибка по фазе
pll_f0    = 8   * 1/3;                                                            # начальная ошибка по частоте
pll_accel0= 0.1 * 0;                                                            # начальная ошибка по ускорению
pll_snr0  = 30;                                                                 # начальная оценка С/Ш (НЕ ошибка)
D0 = np.matrix( [ [0.5**2, 0    , 0    ,    0],
                  [0     , 10**2, 0    ,    0],
                  [0     , 0    , 10**2,    0],
                  [0     , 0    , 0    , 9**2] ])
""" параметры реального сигнала """
T_end = 60;
T_accel = [15, 45]
F_accel = 1 / 20
A_accel = 0; 
SNR   = [500, 350, 400]
T_snr = [10, 30]
"""============================================= 
   Далее производятся расчёты по исходным данным
   ============================================="""
"""------- calc other params ------------"""
T_acc = n_acc * T0;
N_point = int(T_end / T_acc);
time = np.arange(0, N_point, 1) * T_acc
"""-----------------------------------------------"""
""" медель реального сигнала (входное воздействие)"""
""" SNR """
N_snr = [int(T / T_acc) for T in T_snr ]
snr = np.array([SNR[0]] * N_point, float)
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
rva[2, N_accel[0] : N_accel[1]] = A_accel * np.sin( 2 * np.pi * F_accel * t_accel);
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

Dfn = np.matrix( '1 0; 0 1' )
D1  = np.matrix( [[float('nan')] * 4] * 4 ) 
X   = np.matrix( [[float('nan')] * N_point] * 4 )
eye = np.matrix( [ [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1] ], float)
с  = np.matrix( [ [1, 0, 0, 0], 
                  [0, 0, 0, 1] ], float)

X[0, 0] = rva[0, 0] - pll_ph0;
X[1, 0] = rva[1, 0] - pll_f0;
X[2, 0] = rva[0, 0] - pll_accel0;
X[3, 0] = pll_snr0;


tk     = np.arange(0, n_acc) * T0;
tk2to2 = 0.5 * tk ** 2;

for k in range(1, N_point):
    # Входной сигнал 
    sko_y = np.sqrt(0.5) * (10.0 ** (- snr[k] / 20))
    d_ph = rva[0, k - 1] - X[0, k - 1]
    d_fd = rva[1, k - 1] - X[1, k - 1]
    d_ac = rva[2, k - 1] - X[2, k - 1]
    sk = np.exp( j_2pi * (d_ph + d_fd * tk + d_ac * tk2to2)  )
    y = sk + sko_y * (rmd.random(n_acc) + 1j * rmd.random(n_acc))
    # Дискриминатор
    I = y.real
    Q = y.imag
    A = np.sqrt(I**2 + Q**2) 
    Isum = I.sum()
    Qsum = Q.sum()
    Asum = A.sum()
    ph_discr = np.sign(Isum) * Qsum / (2 * np.pi * Asum)
    
    X[0, k] = X[0, k - 1] + ph_discr
    X[1, k] = X[1, k - 1] + ph_discr / T_acc * 8e-4
    X[2, k] = X[2, k - 1] + ph_discr * 2 / T_acc**2 * 8e-5
    
    #X[0, k] += X[1, k] * n_acc / 2
    
    #print('{}; {}; {}'.format(ph_discr, ph_discr / T_acc, ph_discr * 2 / T_acc**2))
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

plt.figure(2)
plt.clf()
plt.subplot(4,1,1)
plt.title('ошибки измерений')
plt.plot(time, (rva[0, : ] - X[0, : ]).A1)
plt.grid(), plt.xlabel('t, sec'), plt.ylabel('ph, cycles')

plt.subplot(4,1,2)
plt.plot(time, (rva[1, : ] - X[1, : ]).A1)
plt.grid(), plt.xlabel('t, sec'), plt.ylabel('fd, Hz')

plt.subplot(4,1,3)
plt.plot(time, (rva[2, : ] - X[2, : ]).A1)
plt.grid(), plt.xlabel('t, sec'), plt.ylabel('accel, Hz/sec')

plt.subplot(4,1,4)
plt.plot(time, snr - X[3, : ].A1)
plt.grid(), plt.xlabel('t, sec'), plt.ylabel('SNR, dBHz')



