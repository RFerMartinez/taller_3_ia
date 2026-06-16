import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética profesional
sns.set_theme(style="whitegrid")

df = pd.read_csv('dataset_delivery.csv')

print("=== 1. TIPOS DE DATOS ORIGINALES ===")
print(df.dtypes)

# === 2. TRATAMIENTO DE VALORES FALTANTES (NaN) ===
cols_numericas = ['distancia_km', 'repartidores_activos', 'tiempo_entrega_min']
for col in cols_numericas:
    mediana = df[col].median()
    df[col] = df[col].fillna(mediana)

cols_categoricas = ['clima', 'zona']
for col in cols_categoricas:
    moda = df[col].mode()[0]
    df[col] = df[col].fillna(moda)

print("\n✅ Valores faltantes procesados. NaNs restantes:", df.isna().sum().sum())

# === 3. TRATAMIENTO DE OUTLIERS (MÉTODO IQR) ===
df_antes_outliers = df.copy()

def detectar_remover_outliers(data, columna):
    Q1 = data[columna].quantile(0.25)
    Q3 = data[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = data[(data[columna] < limite_inferior) | (data[columna] > limite_superior)]
    print(f" -> Outliers detectados en '{columna}': {len(outliers)}")
    return data[(data[columna] >= limite_inferior) & (data[columna] <= limite_superior)]

print("\n=== LIMPIANDO OUTLIERS ===")
df = detectar_remover_outliers(df, 'tiempo_entrega_min')
df = detectar_remover_outliers(df, 'distancia_km')
df = detectar_remover_outliers(df, 'cantidad_productos')
df = detectar_remover_outliers(df, 'repartidores_activos')

# === GRÁFICO COMPARATIVO MEJORADO ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Se eliminó el parámetro y=1.02 para evitar que el título se suba demasiado
fig.suptitle('Impacto de la Remoción de Outliers (Método IQR)', fontsize=16, fontweight='bold')

flierprops_red = dict(marker='o', markerfacecolor='#e74c3c', markersize=9, markeredgecolor='black', linestyle='none')

# Gráfico 1: Antes (Tono Rojo de advertencia)
sns.boxplot(y=df_antes_outliers['tiempo_entrega_min'], ax=axes[0], color='#ffb8b8', width=0.4, flierprops=flierprops_red, linewidth=1.5)
# Agregamos los puntos reales de fondo (stripplot) para que se vea exactamente dónde están los pedidos
sns.stripplot(y=df_antes_outliers['tiempo_entrega_min'], ax=axes[0], color='black', alpha=0.3, jitter=True, size=4)
axes[0].set_title('❌ Antes: Con Outliers Extremos', fontsize=14, color='#c0392b', fontweight='bold')
axes[0].set_ylabel('Tiempo de entrega (minutos)', fontsize=12)

# Gráfico 2: Después (Tono Verde de aprobado)
sns.boxplot(y=df['tiempo_entrega_min'], ax=axes[1], color='#b8ffc8', width=0.4, linewidth=1.5)
sns.stripplot(y=df['tiempo_entrega_min'], ax=axes[1], color='black', alpha=0.5, jitter=True, size=5)
axes[1].set_title('✅ Después: Datos Limpios', fontsize=14, color='#27ae60', fontweight='bold')
axes[1].set_ylabel('Tiempo de entrega (minutos)', fontsize=12)

sns.despine() 
plt.tight_layout()
# Se ajusta el margen superior de la figura para que el suptitle respire
fig.subplots_adjust(top=0.88)
plt.show()

# === 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS ===
df_procesado = pd.get_dummies(df, columns=['clima', 'zona'], drop_first=False)

# === 5. CASTEO DE TIPOS ===
df_procesado['cantidad_productos'] = df_procesado['cantidad_productos'].astype(int)
df_procesado['repartidores_activos'] = df_procesado['repartidores_activos'].astype(int)
df_procesado['hora_dia'] = df_procesado['hora_dia'].astype(int)

for col in df_procesado.columns:
    if df_procesado[col].dtype == bool:
        df_procesado[col] = df_procesado[col].astype(int)

print("\n=== TIPOS DE DATOS DESPUÉS DEL PREPROCESAMIENTO ===")
print(df_procesado.dtypes)

df_procesado.to_csv('dataset_delivery_limpio.csv', index=False)
print("\n✅ Dataset limpio y guardado como 'dataset_delivery_limpio.csv'")