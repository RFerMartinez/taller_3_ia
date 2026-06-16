import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración estética súper profesional
sns.set_theme(style="ticks")
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 1. Cargar el dataset
df = pd.read_csv('dataset_delivery.csv')

print("--- INFORMACIÓN DEL DATASET ---")
df.info()
print("\n--- MÉTRICAS ESTADÍSTICAS ---")
print(df.describe())

# 2. Gráficos Exploratorios
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Análisis Exploratorio Inicial de Delivery (Con Outliers)', fontsize=18, fontweight='bold', y=1.05)

# A. Histograma (Distribución)
# Añadimos bordes negros (edgecolor) y una paleta azul moderna
sns.histplot(df['tiempo_entrega_min'].dropna(), bins=20, kde=True, ax=axes[0], color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.7)
axes[0].set_title('Distribución del Tiempo de Entrega', fontweight='bold')
axes[0].set_xlabel('Minutos de Entrega')
axes[0].set_ylabel('Frecuencia (Cantidad de Pedidos)')
sns.despine(ax=axes[0]) # Quita los bordes superior y derecho para dar respiro visual

# B. Boxplot (Detección visual de Outliers)
# Resaltamos los outliers con un color rojo llamativo y hacemos la caja más delgada
flierprops = dict(marker='o', markerfacecolor='#e74c3c', markersize=9, markeredgecolor='black', linestyle='none')
sns.boxplot(y=df['tiempo_entrega_min'], ax=axes[1], color='#2ecc71', width=0.35, linewidth=2, flierprops=flierprops)
axes[1].set_title('Identificación de Outliers (Boxplot)', fontweight='bold')
axes[1].set_ylabel('Tiempo de entrega (minutos)')
sns.despine(ax=axes[1])

# C. Heatmap de Correlación (Solo variables numéricas)
df_numerico = df.select_dtypes(include=['float64', 'int64'])
correlacion = df_numerico.corr()

# Máscara para ocultar el triángulo superior (técnica muy profesional en Data Science)
mask = np.triu(np.ones_like(correlacion, dtype=bool))

sns.heatmap(correlacion, annot=True, mask=mask, cmap='RdBu_r', fmt=".2f", 
            linewidths=1.5, ax=axes[2], vmin=-1, vmax=1, 
            cbar_kws={"shrink": .8, "label": "Nivel de Correlación"},
            annot_kws={"size": 11, "weight": "bold"})
axes[2].set_title('Matriz de Correlación Lineal', fontweight='bold')
# Rotamos un poco los textos de abajo para que no se superpongan
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()