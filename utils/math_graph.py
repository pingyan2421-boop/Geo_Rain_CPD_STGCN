# @Time     : Jan. 10, 2019 15:21
# @Author   : Veritas YIN
# @FileName : math_graph.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np
import os
import pandas as pd
from scipy.sparse.linalg import eigs


def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def weight_matrix(file_path, n_route=None, mode='auto'):
    """
    修复版 weight_matrix（强健版本）

    mode:
        'auto'      -> 有文件就读，没有就自动生成（推荐）
        'identity'  -> 单位矩阵
        'random'    -> 随机图
    """

    # ✅ 情况1：文件存在
    if os.path.exists(file_path):
        print("✅ Loading adjacency matrix:", file_path)
        W = np.loadtxt(file_path, delimiter=',')

    # ❗情况2：文件不存在（你现在就是这个）
    else:
        print("⚠️ Adjacency file NOT found:", file_path)

        if n_route is None:
            raise ValueError("n_route must be provided when file is missing")

        # 👉 推荐默认：单位矩阵（先跑通）
        print("👉 Using Identity Matrix as fallback")
        W = np.eye(n_route)

        # 你也可以换：
        # W = np.random.rand(n_route, n_route)

    # ✅ 防止 NaN
    W = np.nan_to_num(W)

    # ✅ 对称化（很重要）
    W = (W + W.T) / 2

    # ✅ 归一化（STGCN常用）
    if np.max(W) > 0:
        W = W / np.max(W)

    return W