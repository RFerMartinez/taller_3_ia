# Taller 3: Machine Learning con Python 🤖📈

Este repositorio contiene el desarrollo del **Taller 3** para la asignatura **Inteligencia Artificial (Año 2026)** de la carrera *Ingeniería en Sistemas de Información*. El objetivo principal del taller es comprender los fundamentos matemáticos de Machine Learning implementándolos desde cero con NumPy, aplicar técnicas de preprocesamiento de datos, y entrenar, evaluar y comparar modelos de Regresión Lineal y Árboles de Decisión utilizando Scikit-Learn.

---

## 🚀 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en tu sistema:
* **Python 3.10** o superior.
* **Git** (para clonar el repositorio).

---

## 🛠️ Instalación y Configuración del Entorno

Sigue estos pasos detallados para descargar el repositorio, configurar un entorno virtual (`venv`) e instalar todas las dependencias necesarias de manera aislada.

### 1. Clonar el Repositorio
Abre tu terminal o símbolo del sistema (`cmd`) y ejecuta el siguiente comando para clonar el proyecto:

```bash
git clone [https://github.com/rfermartinez/taller_3_ia.git](https://github.com/rfermartinez/taller_3_ia.git)
```

Luego, accede a la carpeta del proyecto:
```bash
cd taller_3_ia
```

### 2. Crear el Entorno Virtual (`venv`)
Es una buena práctica crear un entorno virtual para no generar conflictos con otras librerías globales de tu sistema. Para crearlo, ejecuta:

```bash
python -m venv venv
```

### 3. Activar el Entorno Virtual
Dependiendo del sistema operativo que utilices, ejecuta el comando correspondiente para activarlo:

* **En Windows (Command Prompt - `cmd`):**
    ```cmd
    venv\Scripts\activate
    ```
* **En Windows (PowerShell):**
    ```powershell
    venv\Scripts\Activate.ps1
    ```
* **En Linux / macOS:**
    ```bash
    source venv/bin/activate
    ```

*Una vez activado, verás que el prefijo `(venv)` aparece al inicio de la línea de comandos en tu terminal.*

### 4. Instalar las Dependencias
Con el entorno virtual activado, instala todas las librerías requeridas (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn y sus dependencias internas) especificadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🧪 Verificación del Entorno

Para cumplir con la consigna de **Configuración del entorno** y verificar que todas las librerías esenciales se instalaron correctamente, puedes iniciar un entorno de Python interactivo o ejecutar un script con el siguiente código para tomar tu captura de pantalla obligatoria:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("✅ Entorno configurado correctamente:")
print(f" -> NumPy: {np.__version__}")
print(f" -> Pandas: {pd.__version__}")
print(f" -> Scikit-Learn: {sklearn.__version__}")
```

---

## 📂 Estructura de Contenidos del Taller

El desarrollo del informe y los scripts se dividen en tres grandes bloques según el programa académico:
1. **Parte 1: Fundamentos Matemáticos con NumPy:** Implementación analítica de la hipótesis de regresión lineal y resolución de la Ecuación Normal de forma matricial (`np.linalg.inv` y `np.linalg.pinv`).
2. **Parte 2: Preprocesamiento de Datos:** Limpieza de un dataset `.csv` propio con tratamiento de valores faltantes (NaN), detección y acotamiento de *outliers* (IQR), y codificación de variables categóricas.
3. **Parte 3: Entrenamiento y Comparación de Modelos:** Evaluación comparativa mediante métricas ($MSE$ y $R^2$) de un modelo de `LinearRegression` frente a un `DecisionTreeRegressor`.
