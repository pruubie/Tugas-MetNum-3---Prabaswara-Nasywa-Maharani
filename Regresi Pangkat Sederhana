import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data Jumlah Latihan Soal (NL) dan Nilai Ujian (NT)
NL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Jumlah Latihan Soal
NT = np.array([52, 55, 60, 63, 65, 70, 72, 75, 80, 85])  # Nilai Ujian

# Fungsi model pangkat sederhana
def power_model(x, a, b):
    return a * np.power(x, b)

# Melakukan fitting pada model pangkat sederhana
popt, pcov = curve_fit(power_model, NL, NT)
a, b = popt

# Memprediksi nilai NT menggunakan model pangkat sederhana
NT_pred_pangkat = power_model(NL, a, b)

# Plot hasil regresi pangkat sederhana
plt.figure(figsize=(10, 6))
plt.scatter(NL, NT, color='red', label='Data Aktual')
plt.plot(NL, NT_pred_pangkat, color='blue', label='Regresi Pangkat Sederhana')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.title('Regresi Pangkat Sederhana Jumlah Latihan Soal vs Nilai Ujian')
plt.grid(True)
plt.show()

# Menampilkan koefisien regresi pangkat sederhana
print(f"Koefisien regresi pangkat sederhana: a = {a}, b = {b}")
