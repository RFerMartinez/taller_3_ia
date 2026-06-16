import pandas as pd
import numpy as np

# Fijamos semilla para que el dataset sea reproducible
np.random.seed(42)

n_registros = 50

# Generamos variables numéricas con distribuciones lógicas
distancia_km = np.round(np.random.uniform(1.0, 15.0, n_registros), 1)
cantidad_productos = np.random.randint(1, 8, n_registros)
hora_dia = np.random.randint(11, 24, n_registros) # De 11 AM a 11 PM
repartidores_activos = np.random.randint(5, 30, n_registros)

# Generamos variables categóricas
climas = ['Despejado', 'Lluvioso', 'Nublado']
clima = np.random.choice(climas, n_registros, p=[0.6, 0.2, 0.2])

zonas = ['Centro', 'Norte', 'Sur', 'Oeste']
zona = np.random.choice(zonas, n_registros)

# Generamos el Target (Tiempo de entrega en minutos) 
# Le damos una relación lógica: más distancia y productos = más tiempo. Más repartidores = menos tiempo.
tiempo_entrega = 15 + (distancia_km * 3) + (cantidad_productos * 2) - (repartidores_activos * 0.5)
# Si llueve, sumamos 10 minutos
tiempo_entrega = np.where(clima == 'Lluvioso', tiempo_entrega + 10, tiempo_entrega)
# Agregamos un poco de ruido aleatorio
tiempo_entrega += np.random.normal(0, 3, n_registros)
tiempo_entrega_min = np.round(np.maximum(10, tiempo_entrega), 0) # Mínimo 10 min

# Creamos el DataFrame
df = pd.DataFrame({
    'distancia_km': distancia_km,
    'cantidad_productos': cantidad_productos,
    'hora_dia': hora_dia,
    'clima': clima,
    'zona': zona,
    'repartidores_activos': repartidores_activos,
    'tiempo_entrega_min': tiempo_entrega_min
})

# --- INTRODUCCIÓN DE RUIDO INTENCIONAL (CONSIGNA 2.1) ---

# 1. Introducimos 5 valores faltantes (NaN) en distintas columnas
indices_nan = np.random.choice(df.index, 5, replace=False)
df.loc[indices_nan[0], 'distancia_km'] = np.nan
df.loc[indices_nan[1], 'clima'] = np.nan
df.loc[indices_nan[2], 'repartidores_activos'] = np.nan
df.loc[indices_nan[3], 'tiempo_entrega_min'] = np.nan
df.loc[indices_nan[4], 'zona'] = np.nan

# 2. Introducimos 4 Outliers muy evidentes
indices_outliers = np.random.choice(np.setdiff1d(df.index, indices_nan), 4, replace=False)
df.loc[indices_outliers[0], 'distancia_km'] = 550.0       # Outlier: 550 km en delivery
df.loc[indices_outliers[1], 'tiempo_entrega_min'] = 720.0 # Outlier: 12 horas de demora
df.loc[indices_outliers[2], 'cantidad_productos'] = 150   # Outlier: 150 hamburguesas
df.loc[indices_outliers[3], 'repartidores_activos'] = -5  # Outlier: valor negativo imposible

# Exportamos a CSV
ruta_csv = 'dataset_delivery.csv'
df.to_csv(ruta_csv, index=False)
print(f"✅ Dataset original creado y guardado en: {ruta_csv}")