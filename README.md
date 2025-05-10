
# **Proyecto de Clasificación de Condición de Productos (Nuevo vs Usado)**

## **1. Introducción**

Este proyecto tiene como objetivo la **clasificación de productos** en dos categorías: **nuevo** y **usado** (`condition: new / used`). El modelo se entrena utilizando un dataset que contiene información sobre productos, incluyendo detalles como **precio**, **cantidad vendida**, **métodos de pago disponibles**, **envío gratuito** y más. El proceso se divide en dos partes principales:

1. **Análisis Exploratorio de Datos (EDA)**: Evaluación preliminar del dataset para entender su estructura, distribuciones y relaciones entre variables.
2. **Entrenamiento de Modelos de Clasificación**: Desarrollo de modelos de aprendizaje automático para predecir la condición de los productos.

Este README cubre ambos pasos, explicando los procesos, las herramientas utilizadas, los análisis realizados y los resultados obtenidos.

---

## **2. Requerimientos del Proyecto**

Para ejecutar este proyecto, necesitarás las siguientes bibliotecas de Python:

- `pandas` para manipulación de datos.
- `numpy` para operaciones numéricas.
- `matplotlib` y `seaborn` para visualización de datos.
- `scikit-learn` para entrenamiento y evaluación de modelos.
- `xgboost` para el modelo de XGBoost.

Las bibliotecas se instalan a partir del archivo **`requirements.txt`** que está ubicado en la ruta **`../source/`**. Para instalar las dependencias, ejecuta el siguiente comando:

```
instalar el entorno con pip install ipykernel y reiniciar visual en caso de no salga el venv

pip install -r source/requirements.txt
```
Se requiere poner el MlA_100k.jsonlines en la carpeta de source antes de correr el notebook del EDA
---

## **3. Análisis Exploratorio de Datos (EDA)**

### **3.1. Carga del Dataset**

El dataset contiene varias columnas con diferentes tipos de datos. Al principio, cargamos el dataset y verificamos sus características, incluyendo los tipos de datos de las columnas, los valores nulos y las distribuciones.

```python
df = pd.read_csv('ruta_del_dataset.csv')
```

### **3.2. Limpieza de Datos**

En esta etapa, eliminamos columnas irrelevantes que no aportan información útil para la clasificación. Las columnas eliminadas incluyen `id`, `title`, `local_pick_up`, `latitude` y `longitude`.

```python
df_cleaned = clean_columns(df)
```

### **3.3. Análisis de Tipos de Datos y Valores Únicos**

Para entender mejor los datos, verificamos los tipos de datos de cada columna y los valores únicos en las columnas categóricas.

```python
df.dtypes
df.nunique()
```

### **3.4. Visualización de Datos**

Varias gráficas fueron generadas para explorar las relaciones y distribuciones de las variables:

1. **Distribución de la variable `condition`** (usando un gráfico de barras):

   Este gráfico muestra la cantidad de productos en cada categoría (nuevo vs. usado), lo que ayuda a entender si el dataset está equilibrado entre las dos clases.

   ```python
   sns.countplot(x='condition', data=df)
   ```

2. **Matriz de Correlación**: La matriz de correlación ayuda a identificar la relación entre las variables numéricas. Se generó un **mapa de calor** para mostrar estas correlaciones.

   ```python
   correlation_matrix = df[numerical_columns].corr()
   sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
   ```

3. **Boxplot de `price`**: Para detectar **outliers** en los valores de precio, se utilizó un boxplot.

   ```python
   sns.boxplot(x=df['price'])
   ```

4. **Gráfico de dispersión entre `price` y `base_price`**: Se generó un gráfico de dispersión para visualizar la relación entre el precio y el precio base de los productos.

   ```python
   sns.scatterplot(x=df['price'], y=df['base_price'])
   ```

### **3.5. Resultados del EDA**

Las visualizaciones y el análisis preliminar mostraron lo siguiente:

- La **distribución de `condition`** mostró que las clases `new` y `used` están bastante equilibradas.
- La **matriz de correlación** reveló que hay relaciones fuertes entre algunas variables numéricas (como `price` y `base_price`), lo que podría ayudar en el modelo.
- El **boxplot de `price`** mostró la presencia de **outliers** con precios extremadamente altos. Esto es importante para considerar cómo manejar esos valores antes de entrenar el modelo.
- El **gráfico de dispersión** entre `price` y `base_price` mostró una relación clara entre ambas variables, con algunos valores atípicos.

---

## **4. Preprocesamiento de Datos**

### **4.1. Transformación de Variables Categóricas**

Se codificaron las variables categóricas (`condition`, `shipping_mode`, `accepts_mercadopago`, etc.) utilizando técnicas de **Label Encoding** y **Categorical Encoding** para convertirlas en variables numéricas.

```python
df_cleaned = encode_condition(df_cleaned)
df_cleaned = encode_categorical_columns(df_cleaned)
```

### **4.2. Escalado de Características Numéricas**

Las características numéricas fueron escaladas usando **StandardScaler** para normalizar las variables.

```python
df_cleaned = scale_features(df_cleaned, numerical_columns)
```

### **4.3. Dividir Datos en Características y Objetivo**

El conjunto de datos se dividió en las características **X** y la variable objetivo **y**.

```python
X = df_cleaned.drop(columns=['condition'])
y = df_cleaned['condition']
```

---

## **5. Entrenamiento de Modelos de Clasificación**

### **5.1. Modelos Utilizados**

Tres modelos de clasificación fueron entrenados y evaluados:

1. **Regresión Logística**
2. **Random Forest**
3. **XGBoost**

### **5.2. Entrenamiento y Evaluación** 

Para cada modelo, se entrenó con un conjunto de entrenamiento (80%) y se evaluó en un conjunto de prueba (20%). Las métricas utilizadas para evaluar los modelos fueron:

- **Matriz de Confusión**
- **Precision, Recall, F1-Score** (Reporte de clasificación)

### **5.3. Resultados de los Modelos**

**Regresión Logística**:
- **Accuracy**: 0.70
- **F1-Score**: 0.70
- **Matriz de Confusión**:
  [[6718 3999],
   [2080 7203]]

**Random Forest**:
- **Accuracy**: 0.82
- **F1-Score**: 0.82
- **Matriz de Confusión**:
  [[8449 2268],
   [1363 7920]]

**XGBoost**:
- **Accuracy**: 0.82
- **F1-Score**: 0.82
- **Matriz de Confusión**:
  [[8339 2378],
   [1178 8105]]

### **5.4. Conclusiones de los Modelos**

- **XGBoost** y **Random Forest** tienen un desempeño similar y son significativamente mejores que **Regresión Logística** en términos de **accuracy** y **f1-score**.
- **Random Forest** y **XGBoost** son los modelos recomendados para este conjunto de datos, con **XGBoost** mostrando una ligera ventaja en la **precisión** para la clase **`new`**.

---

## **6. Conclusiones Generales**

- El **EDA** proporcionó una visión clara de las relaciones entre las variables y la distribución de las clases. Fue crucial para la decisión de preprocesar y manejar **outliers** en el precio.
- Los modelos de **Random Forest** y **XGBoost** son adecuados para este tipo de clasificación, mientras que la **Regresión Logística** no fue tan efectiva.
- Se recomienda realizar un **ajuste de parámetros** (como el ajuste de hiperparámetros) para mejorar aún más los modelos **Random Forest** y **XGBoost**.

---

Este README cubre el proceso completo de análisis, preprocesamiento y entrenamiento de modelos. Si necesitas más detalles o ajustes en alguna sección, no dudes en preguntar.
