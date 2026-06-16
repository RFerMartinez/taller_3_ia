import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams['axes.titlesize'] = 14

# =============================================================
# 3.1 PREPARACIÓN PARA EL MODELADO
# =============================================================

# Cargar dataset limpio
df = pd.read_csv('dataset_delivery_limpio.csv')

# Definir variables predictoras (X) y objetivo (y)
# tiempo_entrega_min es el indicador clave de rendimiento del sistema de delivery
X = df.drop(columns=['tiempo_entrega_min'])
y = df['tiempo_entrega_min']

# División 70% entrenamiento / 30% prueba con semilla fija para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarización de features con StandardScaler
# Se aplica fit_transform SOLO en train y transform en test para evitar data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== 3.1 PREPARACIÓN ===")
print(f"Tamaño entrenamiento: {X_train.shape[0]} registros")
print(f"Tamaño prueba:        {X_test.shape[0]} registros")
print(f"Features utilizadas:  {list(X.columns)}")

# =============================================================
# 3.2 MODELO 1: REGRESIÓN LINEAL
# =============================================================
print("\n=== MODELO 1: REGRESIÓN LINEAL ===")

modelo_lr = LinearRegression()
modelo_lr.fit(X_train_scaled, y_train)

# Predicciones y métricas sobre conjunto de prueba
y_pred_lr = modelo_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MSE: {mse_lr:.2f}")
print(f"R²:  {r2_lr:.4f}")

# Coeficientes del modelo
coeficientes = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': modelo_lr.coef_}
).sort_values(by='Coeficiente', ascending=False)

print("\nCoeficientes del modelo (ordenados de mayor a menor):")
print(coeficientes.to_string(index=False))
print(f"\nIntercepto (θ₀): {modelo_lr.intercept_:.4f}")

# Gráfico: Valores Reales vs Predichos — Regresión Lineal
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='#3498db', edgecolors='k', label='Predicciones')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Diagonal ideal')
plt.xlabel("Tiempo Real (min)")
plt.ylabel("Tiempo Predicho (min)")
plt.title("Regresión Lineal: Valores Reales vs Predichos")
plt.legend()
plt.tight_layout()
plt.savefig('lr_reales_vs_predichos.png', dpi=150)
plt.show()

# =============================================================
# 3.3 MODELO 2: ÁRBOL DE DECISIÓN
# =============================================================
print("\n=== MODELO 2: ÁRBOL DE DECISIÓN ===")

# max_depth=4 para capturar relaciones no lineales sin sobreajustar
modelo_dt = DecisionTreeRegressor(max_depth=4, random_state=42)
modelo_dt.fit(X_train_scaled, y_train)

# Predicciones y métricas sobre conjunto de prueba
y_pred_dt = modelo_dt.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"MSE: {mse_dt:.2f}")
print(f"R²:  {r2_dt:.4f}")

# Importancia de features
importancias = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': modelo_dt.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\nImportancia de las variables (ordenadas de mayor a menor):")
print(importancias.to_string(index=False))

# Visualización del árbol
plt.figure(figsize=(20, 10))
plot_tree(
    modelo_dt,
    feature_names=list(X.columns),   # lista explícita para que muestre los nombres
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Estructura del Árbol de Decisión (max_depth=4)")
plt.tight_layout()
plt.savefig('arbol_decision.png', dpi=150)
plt.show()

# Gráfico: Valores Reales vs Predichos — Árbol de Decisión
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.7, color='#e67e22', edgecolors='k', label='Predicciones')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Diagonal ideal')
plt.xlabel("Tiempo Real (min)")
plt.ylabel("Tiempo Predicho (min)")
plt.title("Árbol de Decisión: Valores Reales vs Predichos")
plt.legend()
plt.tight_layout()
plt.savefig('dt_reales_vs_predichos.png', dpi=150)
plt.show()

# Gráfico: Importancia de Features
plt.figure(figsize=(8, 5))
importancias_sorted = importancias.sort_values('Importancia')
plt.barh(importancias_sorted['Feature'], importancias_sorted['Importancia'], color='#2ecc71', edgecolor='k')
plt.xlabel("Importancia relativa")
plt.title("Importancia de Features — Árbol de Decisión")
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150)
plt.show()

# =============================================================
# 3.4 COMPARACIÓN Y CONCLUSIONES
# =============================================================
print("\n=== 3.4 TABLA COMPARATIVA ===")
print(f"{'Métrica':<20} {'Regresión Lineal':>18} {'Árbol de Decisión':>20}")
print("-" * 60)
print(f"{'MSE (test)':<20} {mse_lr:>18.2f} {mse_dt:>20.2f}")
print(f"{'R² (test)':<20} {r2_lr:>18.4f} {r2_dt:>20.4f}")
print(f"{'Interpretabilidad':<20} {'Alta (coeficientes)':>18} {'Media (árbol visual)':>20}")
sobreajuste_lr = "No (modelo simple)"
sobreajuste_dt = "Posible si depth alto"
print(f"{'¿Sobreentrena?':<20} {sobreajuste_lr:>18} {sobreajuste_dt:>20}")

# Determinar qué modelo tuvo mejor desempeño
print("\n--- Conclusión automática ---")
if mse_dt < mse_lr:
    print(f"El Árbol de Decisión obtuvo menor MSE ({mse_dt:.2f} vs {mse_lr:.2f}).")
else:
    print(f"La Regresión Lineal obtuvo menor MSE ({mse_lr:.2f} vs {mse_dt:.2f}).")

if r2_dt > r2_lr:
    print(f"El Árbol de Decisión explicó mejor la varianza (R²={r2_dt:.4f} vs R²={r2_lr:.4f}).")
else:
    print(f"La Regresión Lineal explicó mejor la varianza (R²={r2_lr:.4f} vs R²={r2_dt:.4f}).")