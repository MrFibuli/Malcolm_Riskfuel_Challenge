import pandas as pd
import numpy as np
from numpy import ndarray

from black_scholes import black_scholes_put

S_bound = (0.0, 200.0)
K_bound = (50.0, 150.0)
T_bound = (0.0, 5.0)
r_bound = (0.001, 0.05)
sigma_bound = (0.05, 1.5)


def gen_data(y):
    return np.random.rand(y, 5)


def gen_put_data(y):
    m = gen_data(y)

    S_delta = S_bound[1] - S_bound[0]
    K_delta = K_bound[1] - K_bound[0]
    T_delta = T_bound[1] - T_bound[0]
    r_delta = r_bound[1] - r_bound[0]
    sigma_delta = sigma_bound[1] - sigma_bound[0]

    deltas: ndarray = np.array([S_delta, K_delta, T_delta, r_delta, sigma_delta])
    lower_bound = np.array([S_bound[0], K_bound[0], T_bound[0], r_bound[0], sigma_bound[0]])

    m = m * deltas + lower_bound
    put = black_scholes_put(m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4]).reshape(-1, 1)

    return np.append(m, put, axis=1)


def data_to_csv():
    #xy = gen_put_data(5000)
    #xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])

    #xy_df.to_csv('put_testdata_5k.csv')

    train = gen_put_data(200_000)
    train_df = pd.DataFrame(train, columns=["S", "K", "T", "r", "sigma", "value"])

    train_df.to_csv('put_traindata_200k.csv')
    # validation = gen_put_data(1000)
    #validation_df = pd.DataFrame(validation, columns=["S", "K", "T", "r", "sigma", "value"])
    #validation_df.to_csv('put_vdata_1k.csv')

