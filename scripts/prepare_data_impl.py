import pandas as pd

input_file = r'C:\Users\lenovo\PycharmProjects\STGCN-Landslide-Prediction\dataset\inter228_5241.csv'
output_file = r'C:\Users\lenovo\PycharmProjects\STGCN-Landslide-Prediction\inter228_200.csv'

# 读取数据
df = pd.read_csv(input_file)

# ❗关键：删除前4列（ID + 经度 + 纬度 + 高程）
df = df.iloc[:, 4:]

# ❗转置（变成：时间 × 点）
df = df.T

# ❗保存（无表头无索引）
df.to_csv(output_file, index=False, header=False)

print("处理完成！shape:", df.shape)