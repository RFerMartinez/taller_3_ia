import numpy as np
import matplotlib.pyplot as plt

# === CONTEXTO DEL DATO ===
# Se modela la carga del procesador (%) de un router MikroTik en función de:
#   x1: Cantidad de túneles VPN activos (10 a 100)
#   x2: Tráfico encriptado en Mbps (50 a 500)
# La relacion real es: y = 5 + 0.3*x1 + 0.1*x2 + ruido

np.random.seed(42)
m = 30

x1 = np.random.randint(10, 100, m)   # Túneles VPN activos
x2 = np.random.randint(50, 500, m)   # Tráfico encriptado (Mbps)
y = 5 + 0.3 * x1 + 0.1 * x2 + np.random.randn(m) * 2

print(f"Training set construido: {m} muestras, 2 variables predictoras")
print(f"  x1 (Tuneles VPN): rango [{x1.min()}, {x1.max()}]")
print(f"  x2 (Trafico Mbps): rango [{x2.min()}, {x2.max()}]")
print(f"  y  (CPU %): rango [{y.min():.1f}, {y.max():.1f}]")
print("=" * 60)

# Matriz X con columna de 1s (intercepto)
X = np.c_[np.ones(m), x1, x2]

# =============================================
# RESOLUCION CON np.linalg.inv (inversa exacta)
# theta = (X^T * X)^-1 * X^T * y
# =============================================
theta_inv = np.linalg.inv(X.T @ X) @ X.T @ y

# =============================================
# RESOLUCION CON np.linalg.pinv (pseudo-inversa de Moore-Penrose)
# =============================================
theta_pinv = np.linalg.pinv(X.T @ X) @ X.T @ y

print("\n--- VALORES OBTENIDOS DE theta ---")
print(f"{'Parametro':<14} {'inv':>14} {'pinv':>14}")
print("-" * 44)
nombres = ['t0 (interc.)', 't1 (VPN)', 't2 (Mbps)']
for nombre, val_inv, val_pinv in zip(nombres, theta_inv, theta_pinv):
    print(f"{nombre:<14} {val_inv:>14.8f} {val_pinv:>14.8f}")

# === COMPARACION inv vs pinv ===
diferencia = np.abs(theta_inv - theta_pinv)
print(f"\nDiferencia absoluta maxima entre inv y pinv: {diferencia.max():.2e}")
son_iguales = "Si" if diferencia.max() < 1e-10 else "No"
print(f"Son practicamente identicos? {son_iguales}")
print()
print("NOTA: En este caso los resultados son casi identicos porque X^T*X es una")
print("matriz bien condicionada (invertible sin problemas). La ventaja de pinv")
print("aparece cuando X^T*X es singular o casi singular (columnas colineales),")
print("donde inv() fallaria con un error pero pinv() sigue dando una solucion")
print("valida usando descomposicion SVD. Por eso pinv es mas estable numericamente.")

# === METRICAS DEL MODELO ===
y_pred = X @ theta_inv
mse = np.mean((y - y_pred) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\n--- METRICAS DEL MODELO ---")
print(f"MSE: {mse:.4f}")
print(f"R2:  {r2:.4f}")

# === GRAFICO: Valores reales vs predichos ===
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='#3498db', alpha=0.7, edgecolors='black', s=60,
            label='Predicciones del modelo')

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
         label='Ajuste perfecto (Real = Predicho)')

plt.title('Ecuacion Normal: Valores Reales vs. Predichos (CPU del Router)')
plt.xlabel('Carga de CPU Real (%)')
plt.ylabel('Carga de CPU Predicha (%)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('1.3_reales_vs_predichos.png', dpi=150)
plt.show()

print("\nGrafico guardado como '1.3_reales_vs_predichos.png'")
