import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data Jumlah Latihan Soal (NL) dan Nilai Ujian (NT)
NL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Jumlah Latihan Soal
NT = np.array([52, 55, 60, 63, 65, 70, 72, 75, 80, 85])  # Nilai Ujian

# Reshape data untuk sklearn
NL_reshaped = NL.reshape(-1, 1)

# Membuat dan melatih model regresi linear
linear_model = LinearRegression()
linear_model.fit(NL_reshaped, NT)

# Memprediksi nilai NT menggunakan model linear
NT_pred_linear = linear_model.predict(NL_reshaped)

# Plot hasil regresi linear
plt.figure(figsize=(10, 6))
plt.scatter(NL, NT, color='red', label='Data Aktual')
plt.plot(NL, NT_pred_linear, color='blue', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.title('Regresi Linear Jumlah Latihan Soal vs Nilai Ujian')
plt.grid(True)
plt.show()

# Menampilkan koefisien regresi
print(f"Koefisien regresi linear: a (slope) = {linear_model.coef_[0]}, b (intercept) = {linear_model.intercept_}")
