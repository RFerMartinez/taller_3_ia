import numpy as np
import matplotlib.pyplot as plt

# 1. Construir el training set (m=30, 2 variables predictoras)
np.random.seed(42) # Fijamos semilla para reproducibilidad
m = 30

# x1: Cantidad de túneles VPN activos
x1 = np.random.randint(10, 100, m)
# x2: Tráfico encriptado (Mbps)
x2 = np.random.randint(50, 500, m)

# y: Carga del procesador del router (%) - Generamos los datos con una relación lineal + ruido
y = 5 + 0.3 * x1 + 0.1 * x2 + np.random.randn(m) * 2

# Matriz X: Agregamos la columna de unos (1) al inicio para multiplicar por theta_0
X = np.c_[np.ones(m), x1, x2]

# 2. Resolver con np.linalg.inv
# Ecuación Normal: θ = (X^T * X)^-1 * X^T * y
theta_inv = np.linalg.inv(X.T @ X) @ X.T @ y

# 3. Resolver con np.linalg.pinv (Pseudo-inversa de Moore-Penrose)
theta_pinv = np.linalg.pinv(X.T @ X) @ X.T @ y

print("--- Valores obtenidos de θ ---")
print(f"θ usando inv:  {theta_inv}")
print(f"θ usando pinv: {theta_pinv}")

# 4. Calcular predicciones para el gráfico usando los thetas obtenidos
y_pred = X @ theta_inv

# 5. Graficar valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Predicciones del Modelo')

# Línea de ajuste perfecto (donde el valor real es exactamente igual al predicho)
min_val = min(min(y), min(y_pred))
max_val = max(max(y), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ajuste perfecto (Real = Predicho)')

plt.title("Valores Reales vs. Predichos (Regresión Múltiple de CPU)")
plt.xlabel("Carga de CPU Real (%)")
plt.ylabel("Carga de CPU Predicha (%)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()