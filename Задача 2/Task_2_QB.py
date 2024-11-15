import pyqiopt as pq
import numpy as np
import csv
from scipy.linalg import block_diag
import pandas as pd

adjacency_matrix = pd.read_csv("task-2-adjacency_matrix.csv").set_index("Location")
graph = np.zeros(np.asarray(adjacency_matrix).shape)
A = np.asarray(adjacency_matrix)

for i in range(len(A)):
    for j in range(len(A[0])):
        if A[i][j] == '-':
            graph[i][j] = 5000

for i in range(len(A)):
    for j in range(len(A[0])):
        if A[i][j] != '-':
            graph[i][j] = int(A[i][j])

tickets = pd.read_csv("task-2-nodes.csv")
tickets = np.asarray(tickets)
tickets_2 = np.zeros(len(tickets))

for i in range(len(tickets)):
    tickets_2[i] = tickets[i][1]

def QUBO(G, m, A, B, C):
    K = 15
    # Первое слагаемое гамильтониана
    def first_term(G, K):
        N = G.shape[0]
        F = np.ones([N**2, N**2])
        for i in range(N**2 - 1):
            F[i][i] = 1 - K
        for i in range(N**2 - 1):
            F[i][N**2-1] = K
            F[N**2-1][i] = K
        F[N**2-1][N**2-1] = - K**2
        V = np.zeros([N**2+5, N**2+5])
        for i in range(N**2):
            for j in range(N**2):
                V[i][j] = F[i][j]
        return V
    

        
        
    # Второе слагаемое гамильтониана
    def second_term(G):
        N = G.shape[0]
        F = np.ones([N, N])
        for i in range(N):
            F[i][i] = -1
        M = block_diag(*(F for _ in range(N)))
        V = np.zeros([N**2+5, N**2+5])
        for i in range(N**2):
            for j in range(N**2):
                V[i][j] = M[i][j]
        return V

    # Третье слагаемое гамильтониана
    def third_term(G):
        N = G.shape[0]
        M = np.zeros([N**2, N**2])
        for i in range(N):
            for j in range(N):
                if G[i][j] == 5000:
                    for p in range(N-1):
                        M[N*i + p][N*j + p + 1] = 1
        V = np.zeros([N**2+5, N**2+5])
        for i in range(N**2):
            for j in range(N**2):
                V[i][j] = M[i][j]
        return V



    # Четвертое слагаемое гамильтониана
    def fourth_term(G):
        N = G.shape[0]
        M = np.zeros([N**2, N**2])
        for i in range(N):
            for j in range(N):
                if G[i][j] != 5000 and G[i][j] != 0:
                    for p in range(N-1):
                        M[i*N + p][j*N + p + 1] = G[i][j]
        V = np.zeros([N**2+5, N**2+5])
        for i in range(N**2):
            for j in range(N**2):
                V[i][j] = M[i][j]
        return V
    
    
    def fifth_term(G, m):
        N = G.shape[0]
        V = np.zeros([N**2+5, N**2+5])
        for i in range(N**2):
            V[i][i] = m[i // N - 1]
        for i in range(N**2 + 1, N**2 + 5):
            V[i][i] = 2**(i - N**2 - 1)
        for i in range(N**2):
            for j in range(N**2 + 1, N**2 + 5):
                V[i][j] = -m[i // N - 1]*2**(j - N**2 - 1)
                V[j][i] = -m[i // N - 1]*2**(j - N**2 - 1)
        return V
        
    
    First = first_term(G, 15)
    Second = second_term(G)
    Third = third_term(G) 
    four = fourth_term(G)
    five = fifth_term(G, m)
    return 10*(First + Second + Third) + four + five

def remove_row_col(matrix, i):
    new_matrix = np.delete(matrix, i, axis=0)
    new_matrix = np.delete(new_matrix, i, axis=1)
    return new_matrix

graph_c = remove_row_col(graph, 2)
B = QUBO(graph_c, tickets_2, 1, 1, 1)
sol = pq.solve(B)
sol_best = sol.vector
sol_value = sol.objective

for i in range(len(sol_best)):
    print(sol_best[i])
print(sol_value)
