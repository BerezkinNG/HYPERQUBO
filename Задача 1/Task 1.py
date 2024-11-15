import numpy as np
import pandas as pd
import random
import seaborn as sn
import scipy.linalg
import pyqiopt as pq

p = 10
N, D = 100, 100 # количество акций и количество дней
N_max = 2000
σ_0 = 0.2
M = 1000000

S = pd.read_csv('task.csv') # matrix of costs
S = np.array(S)

#Cov = np.load('cov_matr.npy')

def get_cov_and_aver( r_ij: np.ndarray ):
    N = r_ij.shape[ 1 ]
    D = r_ij.shape[ 0 ]
    cov = np.zeros( ( N, N ), dtype = float )
    meas = np.zeros( (N), dtype = float )
    for j in range( N ):
        for i in range( D ):
            meas[ j ] += r_ij[ i, j ]
    meas /= D - 1
    for j in range( N ):
        for k in range( N ):
            summ = 0
            for i in range( D - 1 ):
                summ += ( r_ij[ i, j ] - meas[ j ] ) * (  r_ij[ i, k ] - meas[ k ] )
            cov[ j, k ] = summ / ( D - 2 )
    return cov


def qubo_form(C1, C2, S, N, D):
    l = np.zeros(p)
    R_ = np.zeros((D, N))
    
    for i in range(p-1):
        l[i] = 2**i
    l[-1] = N_max + 1 - 2**(p-1)

    L = scipy.linalg.block_diag(*(l for x in range(N)))


    for i in range(D-1):
        for j in range(N):
            R_[i][j] = (S[i+1][j] - S[i][j])/S[i][j]
    
    R_D = np.mean(R_, axis = 0)
    #Cov = np.cov(R_)

    Cov = get_cov_and_aver(R_)
    #L.T@((Cov + R_D.T@R_D)/2*M**2 - np.diag(R_D)/M)@L, L, Cov, R_  
    #σ_0**2*np.ones((N, N))
    
    return  C1*M*L.T@((Cov))@L - C2*L.T@S@R_@L, L, Cov, R_            #L.T@(C1*Cov - C2*np.diag(R_D))@L, L, Cov

C1, C2 = 0.1, 100
qubo, L, Cov, R_ = qubo_form(C1, C2, S, N, D)

sol = pq.solve(qubo, number_of_runs = 1, number_of_steps = 100, return_samples=True, verbose=10)

sol_best = sol.vector
sol_value = sol.objective
sol.dict = sol.samples

list = []
sigma = []
r = np.zeros(D)
r_mean_max = 0
sig_r = []
H = []
for k, v in sol.dict.items():
  print(L@v[0])
  P = S@L@v[0]
  for i in range(1, D):
    r[i] = (P[i] - P[i-1])/P[i-1]
  
  r_mean = np.sum(r)/(D-1)
  if r_mean > r_mean_max:
    r_mean_max = r_mean
    list.append(L@v[0])
    sig_r.append(np.sum((r - r_mean)**2)**0.5)
    H.append(v[1])

  sig = np.sum((r - r_mean)**2)**0.5
  sigma.append(np.sum((r - r_mean)**2)**0.5)

#print(list)
print(sigma)
print(r_mean_max)
print(sig_r[-1])
print(list[-1])
print(sol_best)
print(H[-1])