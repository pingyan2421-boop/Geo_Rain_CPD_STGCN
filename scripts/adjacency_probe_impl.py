import numpy as np

A = np.ones((245, 245))  # 全连接（先跑通用）
np.fill_diagonal(A, 1)

np.save("adj_mat.npy", A)

print("邻接矩阵已生成")