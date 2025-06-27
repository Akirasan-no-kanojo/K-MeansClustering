import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file
df = pd.read_csv("khach_hang.csv")
data = df[['ChiTieuTrungBinh', 'TanSuatMuaSam']].values

# Hàm tính khoảng cách Euclid
def euclidean(a, b):
    return np.linalg.norm(a - b)

# Hàm K-Means
def kmeans(X, k=4, max_iters=100):
    np.random.seed(0)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        labels = []
        for point in X:
            distances = [euclidean(point, c) for c in centroids]
            labels.append(np.argmin(distances))
        labels = np.array(labels)

        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(X[np.random.choice(len(X))])
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# Bước 2: Phân cụm
centroids, labels = kmeans(data, k=4)

# Bước 3: Nhập thông tin khách hàng mới
try:
    chi_tieu = float(input("Nhập Chi tiêu trung bình (triệu đồng/tháng): "))
    tan_suat = float(input("Nhập Tần suất mua sắm (lần/tháng): "))
    khach_moi = np.array([chi_tieu, tan_suat])
except ValueError:
    print("ಥ_ಥ Giá trị nhập không hợp lệ. Hãy nhập số.")
    exit()

# Dự đoán cụm cho khách hàng mới
distances = [euclidean(khach_moi, c) for c in centroids]
nhom = np.argmin(distances)
print(f" (●'◡'●) Khách hàng mới thuộc vào cụm số: {nhom}")

# Bước 4: Vẽ đồ thị kết quả
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue', 'purple']
for i in range(4):
    plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1],
                label=f'Cụm {i}', alpha=0.6, c=colors[i])
plt.scatter(centroids[:, 0], centroids[:, 1], 
            marker='x', s=200, c='black', label='Tâm cụm')
plt.scatter(khach_moi[0], khach_moi[1], 
            marker='o', s=150, c='yellow', edgecolors='black', 
            label=f'Khách mới (thuộc cụm {nhom})')
plt.xlabel('Chi tiêu trung bình (triệu đồng)')
plt.ylabel('Tần suất mua sắm (lần/tháng)')
plt.title('Phân cụm khách hàng bằng K-Means')
plt.legend()
plt.grid(True)
plt.show()
 # type: ignore