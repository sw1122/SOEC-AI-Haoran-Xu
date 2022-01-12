'''
@autor: Wei Shuai
Time: 2022.1.10 at ZJU
'''

from tensorflow_core import keras
import numpy as np
from tensorflow.keras.models import load_model
import os

DNA_SIZE = 30
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
N_GENERATIONS = 200
X_BOUND = [0.2, 1]
Y_BOUND = [0.2, 1]
P = 60
T = 1023

S = 1.44/1000
V_BOUND = [0.7, 1.65]
I_BOUND = [38.06, 30309.58]
T_BOUND = [1023, 1123]
xH2O_BOUND = [0.1, 0.9]
SCCM_BOUND = [300, 600]
xH2_BOUND = [0.0002, 0.8752]
xH2CO_BOUND = [0.001, 0.9889]
Q_BOUND = [-1.4588, 6.9686]
nor_T = 0.8 * (T - T_BOUND[0]) / (T_BOUND[1] - T_BOUND[0]) + 0.2

def rmse(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))

folder = "D:\Desktop\AI+GA"
os.chdir(folder)

model = load_model('D:\Desktop\AI+GA\model.h5',custom_objects={'rmse': rmse})

N = 1
global best_pop
global i_last

def F(x, y):
    global i_last
    num = x.shape[0]
    X2 = []
    I_nor = []
    i_last = 0
    xH2_max = 0.1
    SCCM_max = 0
    SCCM_min = 4

    for i in range(num):
        U = (y[i] - 0.2)/0.8 * (V_BOUND[1] - V_BOUND[0]) + V_BOUND[0]
        I = (P / U) / S
        nor_I = 0.8 * (I - I_BOUND[0]) / (I_BOUND[1] - I_BOUND[0]) + 0.2
        I_nor.append(nor_I)
        X = [y[i], nor_I, nor_T, x[i]]
        X2.append(X)
    Xtest = np.array(X2)
    X_test = Xtest.reshape(-1, 4)
    Y_pre = model.predict(X_test)

    for i in range(num):
        if Y_pre[i, N] > 1.0 or Y_pre[i, N] < 0.2:
            Y_pre[i, 0] = 0.0001
            Y_pre[i, N] = 0.0001
        if Y_pre[i, 0] < 0.2 or Y_pre[i, 0] > 1:
            Y_pre[i, 0] = 0.0001
            Y_pre[i, N] = 0.0001
            continue
        if Y_pre[i, N] > xH2_max:
            i_last = i
            SCCM_max = Y_pre[i, 0]
            xH2_max = Y_pre[i, N]
        elif Y_pre[i, N] == xH2_max:
            if Y_pre[i, 0] > SCCM_max:
                SCCM_max = Y_pre[i, 0]
                i_last = i
            elif Y_pre[i, 0] < SCCM_min:
                SCCM_min = Y_pre[i, 0]

    for i in range(num):
        if Y_pre[i, N] < 0.1:
            continue
        if Y_pre[i, N] == xH2_max:
            if SCCM_max == SCCM_min:
                Y_pre[i, N] = 1.5*xH2_max
            else:
                Y_pre[i, N] = ((Y_pre[i, 0] - SCCM_min) / (SCCM_max - SCCM_min) + 1.5) * (xH2_max + 0.01)

    XH2_MAX.append(xH2_max)
    x_H2O.append(Xtest[i_last, 3])
    u_real = (y[i_last] - 0.2) / 0.8 * (V_BOUND[1] - V_BOUND[0]) + V_BOUND[0]
    i_real = (P / u_real) / S
    x_U_real.append(u_real)
    x_I_real.append(i_real)
    x_SCCMMAX.append(SCCM_max)

    return Y_pre[:, N]

def get_fitness(pop):
    global best_pop
    x, y = translateDNA(pop)
    pred = F(x, y)
    best_pop = pop[i_last]
    return pred

def translateDNA(pop):
    x_pop = pop[:, 1::2]
    y_pop = pop[:, ::2]
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

def crossover_and_mutation(pop):
    new_pop = []
    for father in pop:
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)
            child[cross_points:] = mother[cross_points:]
        mutation(child)
        new_pop.append(child)

    return new_pop

def mutation(child):
    for i in range(len(child)):
        if np.random.rand() < MUTATION_RATE:
            child[i] = child[i] ^ 1

def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness) / (fitness.sum()))
    return pop[idx]

if __name__ == "__main__":
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    XH2_MAX = []
    x_H2O = []
    x_SCCMMAX = []
    x_U_real = []
    x_I_real = []
    x = []
    y = []
    for i in range(N_GENERATIONS):
        fitness = get_fitness(pop)
        x, y = translateDNA(pop)
        pop = select(pop, fitness)
        pop[0, :] = best_pop
        pop = np.array(crossover_and_mutation(pop))
        pop[0, :] = best_pop

    xH2O = (x[i_last] - 0.2) / 0.8 * (0.9 - 0.1) + 0.1
    U_last = (y[i_last] - 0.2)/0.8 * (V_BOUND[1] - V_BOUND[0]) + V_BOUND[0]
    I_last = (P / U_last) / S
    nor_I = 0.8 * (I_last - I_BOUND[0]) / (I_BOUND[1] - I_BOUND[0]) + 0.2
    X = [y[i_last], nor_I, nor_T, x[i_last]]
    X = np.array(X)
    XMAX = X.reshape(-1, 4)
    Y_PRE = model.predict(XMAX)
    y_MAX_real = (Y_PRE[0, 0]-0.2)/0.8*(SCCM_BOUND[1]-SCCM_BOUND[0])+SCCM_BOUND[0]
    H2 = Y_PRE[0, N]
    H2_real = (H2 - 0.2) / 0.8 * (xH2_BOUND[1] - xH2_BOUND[0]) + xH2_BOUND[0]

