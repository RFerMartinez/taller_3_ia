import numpy as np
import matplotlib.pyplot as plt

# === CONTEXTO DEL DATO ===
# x = Usuarios concurrentes en VPN PPTP (en decenas)
# y = Consumo de CPU del router MikroTik (%)
# A medida que hay más usuarios conectados al túnel VPN, el router necesita más
# potencia de procesamiento para cifrar/descifrar paquetes, generando una relación lineal.

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_true = np.array([12, 21, 28, 42, 51, 63, 68, 81, 90, 98])

print("Vector x (Usuarios en decenas):", x)
print("Vector y_true (Consumo CPU % real):", y_true)
print("=" * 60)

# Preparar la matriz X para multiplicacion matricial (columna de 1s para theta_0)
m = len(x)
X = np.c_[np.ones(m), x]

def calcular_mse(y_verdadero, y_predicho):
    return np.mean((y_verdadero - y_predicho) ** 2)

# === PROBAR 3 COMBINACIONES DISTINTAS DE theta ===
combinaciones_theta = [
    np.array([0, 10]),
    np.array([5, 8]),
    np.array([2, 9.5]),
]

colores = ['#e74c3c', '#2ecc71', '#9b59b6']
interpretaciones = [
    "Sin offset (t0=0): asume que sin usuarios el CPU esta en 0%. "
    "La pendiente de 10 predice 10% de CPU por cada decena de usuarios. "
    "El modelo sobreestima en valores bajos y subestima en valores altos.",

    "Con offset bajo (t0=5): parte de un consumo base de 5% (procesos del sistema). "
    "La pendiente de 8 es mas conservadora. Subestima sistematicamente, "
    "indicando que la relacion real crece mas rapido que lo que este modelo captura.",

    "Modelo mas ajustado (t0=2, t1=9.5): el offset de 2% representa un consumo "
    "minimo realista del router en idle. La pendiente de 9.5 captura mejor "
    "el crecimiento real del CPU. Es la combinacion con menor MSE.",
]

print("\n--- EVALUACION DE 3 MODELOS ---\n")
resultados = []
for i, theta in enumerate(combinaciones_theta, 1):
    predicciones = X @ theta  # h(x) = θ₀ + θ₁x (multiplicación matricial, sin bucles for)
    mse = calcular_mse(y_true, predicciones)
    resultados.append((theta, predicciones, mse))

    print(f"Modelo {i}: h(x) = {theta[0]} + {theta[1]}*x")
    print(f"  Predicciones: {predicciones}")
    print(f"  MSE: {mse:.2f}")
    print(f"  >> {interpretaciones[i-1]}")
    print()

# === GRÁFICO: Las 3 rectas vs los datos reales ===
plt.figure(figsize=(10, 6))
plt.scatter(x, y_true, color='black', s=80, zorder=5, label='Datos reales (CPU %)')

for i, (theta, pred, mse) in enumerate(resultados):
    plt.plot(x, pred, color=colores[i], linewidth=2,
             label=f'Modelo {i+1}: h(x)={theta[0]}+{theta[1]}x  (MSE={mse:.1f})')

plt.title('Regresion Lineal desde Cero: 3 Hipotesis sobre Carga de CPU vs Usuarios VPN')
plt.xlabel('Usuarios concurrentes VPN (decenas)')
plt.ylabel('Consumo de CPU del Router (%)')

plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('1.2_comparacion_hipotesis.png', dpi=150)
plt.show()

print("Grafico guardado como '1.2_comparacion_hipotesis.png'")
