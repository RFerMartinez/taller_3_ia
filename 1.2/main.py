import numpy as np

# 1. Definir vector x (10 valores) y un vector y_true para poder calcular el MSE
# Contexto: x = Usuarios concurrentes en VPN PPTP (en decenas)
#           y = Consumo de CPU del router MikroTik (%)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_true = np.array([12, 21, 28, 42, 51, 63, 68, 81, 90, 98])

print("Vector x (Usuarios en decenas):", x)
print("Vector y_true (Consumo CPU % real):", y_true)
print("-" * 50)

# 2. Preparar la matriz X para la multiplicación matricial
# Agregamos una columna de 1s para que multiplique a theta_0
m = len(x)
X = np.c_[np.ones(m), x] 

# Función para calcular el MSE
def calcular_mse(y_verdadero, y_predicho):
    return np.mean((y_verdadero - y_predicho) ** 2)

# 3. Probar con 3 combinaciones distintas de theta (theta_0, theta_1)
combinaciones_theta = [
    np.array([0, 10]),   # Modelo 1: theta_0 = 0, theta_1 = 10
    np.array([5, 8]),    # Modelo 2: theta_0 = 5, theta_1 = 8
    np.array([20, 3])    # Modelo 3: theta_0 = 20, theta_1 = 3
]

for i, theta in enumerate(combinaciones_theta, 1):
    # Implementar h(x) = X @ theta (Multiplicación matricial sin bucles for)
    predicciones = X @ theta
    
    # Calcular el MSE
    mse = calcular_mse(y_true, predicciones)
    
    print(f"--- Prueba {i} ---")
    print(f"Theta: θ0 = {theta[0]}, θ1 = {theta[1]}")
    print(f"Ecuación: h(x) = {theta[0]} + {theta[1]}x")
    print(f"Predicciones: {predicciones}")
    print(f"MSE: {mse:.2f}\n")