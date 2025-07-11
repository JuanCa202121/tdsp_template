# Project Charter - Entendimiento del Negocio

## 1. Nombre del Proyecto

**Clasificación de impago de tarjeta de credito de clientes de un banco**

---

## 2. Objetivo del Proyecto

Desarrollar y evaluar un modelo de clasificación que prediga la probabilidad de impago de los clientes de tarjetas de crédito de un banco, utilizando el dataset público “Default of Credit Card Clients” (UCI / Kaggle). El resultado apoyará la toma de decisiones de riesgo al identificar oportunamente clientes con alta probabilidad de incumplimiento. Lo anterior usando herramientas y conceptos propios de las metodologías agiles aplicadas a la ciencia de datos.

- **Fuente de datos**: [Default of Credit Card Clients (Kaggle)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

---


## 3. Alcance del Proyecto


- **Datos:** dataset **Default of Credit Card Clients Dataset**de 30 000 registros y 23 variables.
  - **ID**: Identificador de cada cliente.
  - **LIMIT_BAL**: Monto de crédito otorgado en dólares NTD (dólar taiwanés), incluye crédito individual y familiar/suplementario.
  - **SEX**: Género (1 = masculino, 2 = femenino).
  - **EDUCATION**: Nivel educativo (1 = posgrado, 2 = universidad, 3 = bachillerato, 4 = otros, 5 = desconocido, 6 = desconocido).
  - **MARRIAGE**: Estado civil (1 = casado, 2 = soltero, 3 = otros).
  - **AGE**: Edad en años.
  - **PAY_0**: Estado de pago en septiembre de 2005 (–1 = pago puntual, 1 = retraso de un mes, 2 = retraso de dos meses, …, 8 = retraso de ocho meses, 9 = retraso de nueve meses o más).
  - **PAY_2**: Estado de pago en agosto de 2005 (misma escala).
  - **PAY_3**: Estado de pago en julio de 2005 (misma escala).
  - **PAY_4**: Estado de pago en junio de 2005 (misma escala).
  - **PAY_5**: Estado de pago en mayo de 2005 (misma escala).
  - **PAY_6**: Estado de pago en abril de 2005 (misma escala).
  - **BILL_AMT1**: Monto de la factura en septiembre de 2005 (dólares NTD).
  - **BILL_AMT2**: Monto de la factura en agosto de 2005 (dólares NTD).
  - **BILL_AMT3**: Monto de la factura en julio de 2005 (dólares NTD).
  - **BILL_AMT4**: Monto de la factura en junio de 2005 (dólares NTD).
  - **BILL_AMT5**: Monto de la factura en mayo de 2005 (dólares NTD).
  - **BILL_AMT6**: Monto de la factura en abril de 2005 (dólares NTD).
  - **PAY_AMT1**: Monto del pago anterior en septiembre de 2005 (dólares NTD).
  - **PAY_AMT2**: Monto del pago anterior en agosto de 2005 (dólares NTD).
  - **PAY_AMT3**: Monto del pago anterior en julio de 2005 (dólares NTD).
  - **PAY_AMT4**: Monto del pago anterior en junio de 2005 (dólares NTD).
  - **PAY_AMT5**: Monto del pago anterior en mayo de 2005 (dólares NTD).
  - **PAY_AMT6**: Monto del pago anterior en abril de 2005 (dólares NTD).
  - **default.payment.next.month**: Indicador de impago el mes siguiente (1 = sí, 0 = no).


- **Fases TDSP** alineadas con la rúbrica del curso:
  1. Entendimiento del negocio y carga de datos.
  2. Preprocesamiento y Análisis exploratorio de datos (EDA).
  3. Modelamiento y Extracción de características.
  4. Despliegue (prototipo local/API) y gestión de versiones.
  5. Evaluación final y presentación.
- **Herramientas** a usar:
  - **Git** → versionamiento de código.
  - **DVC** → versionamiento de datos.
  - **MLflow** → seguimiento de experimentos y modelos.
  - Python, pandas, scikit‑learn, Jupyter, FastAPI (opcional para demo).
- **Entregables** por fase:

| Fase | Entregables obligatorios (según rúbrica) |
|------|-------------------------------------------|
| 1 | Marco de proyecto, código de carga de datos, diccionario de datos |
| 2 | Código de preprocesamiento/EDA, resumen estadístico, visualizaciones |
| 3 | Código de extracción de características, notebooks de modelamiento, reporte de línea base y modelo final (MLflow) |
| 4 | Código de despliegue (prototipo), documentación de despliegue y entorno |
| 5 | Código de evaluación, interpretación de resultados, video de presentación |
| 6 | **Entrega final** |

- **Criterios de éxito (baseline):** AUC‑ROC ≥ 0.78, Recall ≥ 0.60 en el dataset de prueba.

---

## 3.1. Excluye:

- Integrar datos adicionales (p. ej. perfiles transaccionales).
- Despliegue productivo en la nube.
- Análisis financiero detallado más allá de métricas de clasificación.

---

## 3.2 Carga de los datos




---


```python
import pandas as pd
```


```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "UCI_Credit_Card.csv"  # nombre exacto

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "uciml/default-of-credit-card-clients-dataset",
  file_path
)

print("First 5 records:", df.head())

```

    /tmp/ipython-input-7-1678825860.py:6: DeprecationWarning: load_dataset is deprecated and will be removed in a future version.
      df = kagglehub.load_dataset(


    First 5 records:    ID  LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \
    0   1    20000.0    2          2         1   24      2      2     -1     -1   
    1   2   120000.0    2          2         2   26     -1      2      0      0   
    2   3    90000.0    2          2         2   34      0      0      0      0   
    3   4    50000.0    2          2         1   37      0      0      0      0   
    4   5    50000.0    1          2         1   57     -1      0     -1      0   
    
       ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  \
    0  ...        0.0        0.0        0.0       0.0     689.0       0.0   
    1  ...     3272.0     3455.0     3261.0       0.0    1000.0    1000.0   
    2  ...    14331.0    14948.0    15549.0    1518.0    1500.0    1000.0   
    3  ...    28314.0    28959.0    29547.0    2000.0    2019.0    1200.0   
    4  ...    20940.0    19146.0    19131.0    2000.0   36681.0   10000.0   
    
       PAY_AMT4  PAY_AMT5  PAY_AMT6  default.payment.next.month  
    0       0.0       0.0       0.0                           1  
    1    1000.0       0.0    2000.0                           1  
    2    1000.0    1000.0    5000.0                           0  
    3    1100.0    1069.0    1000.0                           0  
    4    9000.0     689.0     679.0                           0  
    
    [5 rows x 25 columns]



```python
df.head()
```





  <div id="df-25be396e-9818-4f43-a1dd-5550d863cd58" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-25be396e-9818-4f43-a1dd-5550d863cd58')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-25be396e-9818-4f43-a1dd-5550d863cd58 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-25be396e-9818-4f43-a1dd-5550d863cd58');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d8958759-29fa-40cf-a0ae-6ae73f165481">
      <button class="colab-df-quickchart" onclick="quickchart('df-d8958759-29fa-40cf-a0ae-6ae73f165481')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d8958759-29fa-40cf-a0ae-6ae73f165481 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
df.info()
df.describe()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 25 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   ID                          30000 non-null  int64  
     1   LIMIT_BAL                   30000 non-null  float64
     2   SEX                         30000 non-null  int64  
     3   EDUCATION                   30000 non-null  int64  
     4   MARRIAGE                    30000 non-null  int64  
     5   AGE                         30000 non-null  int64  
     6   PAY_0                       30000 non-null  int64  
     7   PAY_2                       30000 non-null  int64  
     8   PAY_3                       30000 non-null  int64  
     9   PAY_4                       30000 non-null  int64  
     10  PAY_5                       30000 non-null  int64  
     11  PAY_6                       30000 non-null  int64  
     12  BILL_AMT1                   30000 non-null  float64
     13  BILL_AMT2                   30000 non-null  float64
     14  BILL_AMT3                   30000 non-null  float64
     15  BILL_AMT4                   30000 non-null  float64
     16  BILL_AMT5                   30000 non-null  float64
     17  BILL_AMT6                   30000 non-null  float64
     18  PAY_AMT1                    30000 non-null  float64
     19  PAY_AMT2                    30000 non-null  float64
     20  PAY_AMT3                    30000 non-null  float64
     21  PAY_AMT4                    30000 non-null  float64
     22  PAY_AMT5                    30000 non-null  float64
     23  PAY_AMT6                    30000 non-null  float64
     24  default.payment.next.month  30000 non-null  int64  
    dtypes: float64(13), int64(12)
    memory usage: 5.7 MB






  <div id="df-718367d8-f199-4c2f-8381-07dde1fde411" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>...</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15000.500000</td>
      <td>167484.322667</td>
      <td>1.603733</td>
      <td>1.853133</td>
      <td>1.551867</td>
      <td>35.485500</td>
      <td>-0.016700</td>
      <td>-0.133767</td>
      <td>-0.166200</td>
      <td>-0.220667</td>
      <td>...</td>
      <td>43262.948967</td>
      <td>40311.400967</td>
      <td>38871.760400</td>
      <td>5663.580500</td>
      <td>5.921163e+03</td>
      <td>5225.68150</td>
      <td>4826.076867</td>
      <td>4799.387633</td>
      <td>5215.502567</td>
      <td>0.221200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8660.398374</td>
      <td>129747.661567</td>
      <td>0.489129</td>
      <td>0.790349</td>
      <td>0.521970</td>
      <td>9.217904</td>
      <td>1.123802</td>
      <td>1.197186</td>
      <td>1.196868</td>
      <td>1.169139</td>
      <td>...</td>
      <td>64332.856134</td>
      <td>60797.155770</td>
      <td>59554.107537</td>
      <td>16563.280354</td>
      <td>2.304087e+04</td>
      <td>17606.96147</td>
      <td>15666.159744</td>
      <td>15278.305679</td>
      <td>17777.465775</td>
      <td>0.415062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>10000.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>...</td>
      <td>-170000.000000</td>
      <td>-81334.000000</td>
      <td>-339603.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7500.750000</td>
      <td>50000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>2326.750000</td>
      <td>1763.000000</td>
      <td>1256.000000</td>
      <td>1000.000000</td>
      <td>8.330000e+02</td>
      <td>390.00000</td>
      <td>296.000000</td>
      <td>252.500000</td>
      <td>117.750000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15000.500000</td>
      <td>140000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>19052.000000</td>
      <td>18104.500000</td>
      <td>17071.000000</td>
      <td>2100.000000</td>
      <td>2.009000e+03</td>
      <td>1800.00000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22500.250000</td>
      <td>240000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>54506.000000</td>
      <td>50190.500000</td>
      <td>49198.250000</td>
      <td>5006.000000</td>
      <td>5.000000e+03</td>
      <td>4505.00000</td>
      <td>4013.250000</td>
      <td>4031.500000</td>
      <td>4000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30000.000000</td>
      <td>1000000.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>891586.000000</td>
      <td>927171.000000</td>
      <td>961664.000000</td>
      <td>873552.000000</td>
      <td>1.684259e+06</td>
      <td>896040.00000</td>
      <td>621000.000000</td>
      <td>426529.000000</td>
      <td>528666.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-718367d8-f199-4c2f-8381-07dde1fde411')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-718367d8-f199-4c2f-8381-07dde1fde411 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-718367d8-f199-4c2f-8381-07dde1fde411');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-7cc1f4c1-b196-4759-9211-3df118bba853">
      <button class="colab-df-quickchart" onclick="quickchart('df-7cc1f4c1-b196-4759-9211-3df118bba853')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-7cc1f4c1-b196-4759-9211-3df118bba853 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




**Se ha cargado el CSV con éxito y se ha comprobado que todas las columnas están presentes**


## 4. Metodología

Se adoptará el marco **TDSP** (Team Data Science Process) complementado con prácticas Ágiles y las herramientas del curso:

- **Plan** → Backlog en GitHub Projects, sprints semanales.
- **Develop** → Git + DVC para rastrear notebooks, código y datasets.
- **Build** → MLflow para experimentos, métricas y artefactos de modelos.
- **Deploy** → Prototipo local o API FastAPI..
- **Operate** → Métricas de evaluación y reflexiones sobre el modelo resultante. Reflexionando sobre cómo podría cambiar la relación entre variables y el resultado (impago) en el tiempo, y cómo esto afectaría la vigencia del modelo entrenado.

A continuación se muestra el ciclo de vida del proceso TDSP utilizado en el proyecto:

> ![Ciclo de vida TDSP](./Imagenes/metodologia.png)



---



## 5. Cronograma

| Fase TDSP | Entregables clave | Fechas |
|-----------|------------------|--------|
| 1 – Entendimiento & Carga | • Marco del proyecto  <br>• Notebook de carga de datos  <br>• Diccionario de datos | 26 jun → 03 jul |
| 2 – Preprocesamiento & EDA | • Código de limpieza y EDA  <br>• Resumen estadístico y visualizaciones clave | 04 → 11 jul |
| 3 – Modelamiento & Features | • Pipeline de ingeniería de características  <br>• Modelos baseline y mejorados rastreados en MLflow  <br>• Reporte comparativo | 12 → 18 jul |
| 4 – Despliegue (Prototipo) | • Script/API  <br>• Documentación de infraestructura local  <br>• Configuración de DVC | 29 → 25 jul |
| 5 – Evaluación & Entrega final | • Notebook de evaluación en *hold‑out/particiones entrenamiento prueba*  <br>• Interpretación de resultados y limitaciones  <br>• Video (≤ 5 min) con presentación del proyecto | 28 jul |



---

## 6. Equipo del Proyecto

| Nombre | Rol | Responsabilidades |
|--------|-----|------------------|
| Juan Felipe Caro Monroy | Líder del equipo| Dirección general, código, experimentos, presentación |
| Miguel Ángel Naranjo | Líder del proyecto | Guía técnica,  código, experimentos, presentación |
| Edwin David García| Científico de datos| Guía técnica, código, experimentos, presentación |



---

## 7. Presupuesto

Sin presupuesto monetario, se utilizan recursos gratuitos (Kaggle, GitHub, DVC remotos locales, MLflow local).

---

## 8. Stakeholders

- **Profesor(a) del módulo** – Supervisor académico y evaluador.
- **Tutor/monitor del módulo** – Feedback  durante revisiones y evaluador.
- **Cliente hipotetico de Riesgo Bancario**

---



## 9. Aprobaciones




| Nombre | Cargo | Firma | Fecha |
|--------|-------|-------|-------|
| Jorge Eliécer Camargo Mendoza | Docente | ____________________ | ___ / ___ / 2025 |
| Juan Sebastian Malagón Torres | Tutor/Monitor | ____________________ | ___ / ___ / 2025 |

# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.


## Resumen general de los datos

En esta sección se presenta un resumen general de los datos. Se describe el número total de observaciones, variables, el tipo de variables, la presencia de valores faltantes y la distribución de las variables.

## Resumen de calidad de los datos

En esta sección se presenta un resumen de la calidad de los datos. Se describe la cantidad y porcentaje de valores faltantes, valores extremos, errores y duplicados. También se muestran las acciones tomadas para abordar estos problemas.


```python
# Conteo de valores faltantes
missing_count = df.isnull().sum()
# Porcentaje de valores faltantes
missing_percent = (missing_count / len(df)) * 100

missing_df = pd.DataFrame({
    "Valores Faltantes": missing_count,
    "% del Total": missing_percent
}).sort_values(by="Valores Faltantes", ascending=False)

print("📌 Valores faltantes:\n")
print(missing_df)
```

    📌 Valores faltantes:
    
                                Valores Faltantes  % del Total
    ID                                          0          0.0
    LIMIT_BAL                                   0          0.0
    SEX                                         0          0.0
    EDUCATION                                   0          0.0
    MARRIAGE                                    0          0.0
    AGE                                         0          0.0
    PAY_0                                       0          0.0
    PAY_2                                       0          0.0
    PAY_3                                       0          0.0
    PAY_4                                       0          0.0
    PAY_5                                       0          0.0
    PAY_6                                       0          0.0
    BILL_AMT1                                   0          0.0
    BILL_AMT2                                   0          0.0
    BILL_AMT3                                   0          0.0
    BILL_AMT4                                   0          0.0
    BILL_AMT5                                   0          0.0
    BILL_AMT6                                   0          0.0
    PAY_AMT1                                    0          0.0
    PAY_AMT2                                    0          0.0
    PAY_AMT3                                    0          0.0
    PAY_AMT4                                    0          0.0
    PAY_AMT5                                    0          0.0
    PAY_AMT6                                    0          0.0
    default.payment.next.month                  0          0.0



```python
# Número de filas duplicadas
duplicated_rows = df.duplicated().sum()

print(f"\n📌 Filas duplicadas: {duplicated_rows}")

```

    
    📌 Filas duplicadas: 0


## Variable objetivo

En esta sección se describe la variable objetivo. Se muestra la distribución de la variable y se presentan gráficos que permiten entender mejor su comportamiento.


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Renombramos para facilitar análisis (opcional)
target_col = "default.payment.next.month"

# Conteo absoluto y relativo
counts = df[target_col].value_counts()
percent = df[target_col].value_counts(normalize=True) * 100

# Mostrar resumen numérico
summary = pd.DataFrame({"Frecuencia": counts, "Porcentaje": percent.round(2)})
summary.index = ["No Incumplió (0)", "Incumplió (1)"]
print("📊 Distribución de la variable objetivo:\n")
print(summary)

```

    📊 Distribución de la variable objetivo:
    
                      Frecuencia  Porcentaje
    No Incumplió (0)       23364       77.88
    Incumplió (1)           6636       22.12



```python
# Gráfico de barras
plt.figure(figsize=(6,4))
sns.countplot(data=df, x=target_col, palette="Set2")
plt.title("Distribución de la Variable Objetivo")
plt.xlabel("Default en el siguiente mes (0 = No, 1 = Sí)")
plt.ylabel("Cantidad de clientes")
plt.xticks([0, 1], ["No incumplió", "Incumplió"])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

```

    /tmp/ipython-input-14-1027984647.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x=target_col, palette="Set2")



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_30_1.png)
    


La variable objetivo 'default.payment.next.month' es binaria y representa si un cliente incumplió con el pago de su crédito en el siguiente mes.

De los 30.000 registros:
- Alrededor del 77% de los clientes no incumplieron con sus pagos.
- Mientras que el 22% sí lo hicieron.

Esta distribución indica un desbalance moderado entre clases, lo cual puede influir en el desempeño de los modelos de clasificación. Este desbalance deberá tenerse en cuenta al momento de entrenar el modelo (por ejemplo, utilizando métricas como F1-score, o aplicando técnicas como sobre/sobremuestreo o ajuste de pesos).

**Soluciones**

a. Muchos algoritmos de clasificación permiten penalizar más los errores cometidos en la clase minoritaria. Esto se logra ajustando los pesos de las clases en la función de pérdida del modelo:

Se le da mayor peso a la clase menos representada.

Ejemplo: en XGBoost, se usa el parámetro scale_pos_weight = n_negativos / n_positivos.

b. Remuestreo de los datos
Esta técnica implica modificar la muestra de entrenamiento para equilibrar la proporción de clases.

c. Oversampling (sobremuestreo)
Consiste en aumentar artificialmente la clase minoritaria. Puede hacerse replicando ejemplos o generando nuevos (por ejemplo, con SMOTE).


## Variables individuales

En esta sección se presenta un análisis detallado de cada variable individual. Se muestran estadísticas descriptivas, gráficos de distribución y de relación con la variable objetivo (si aplica). Además, se describen posibles transformaciones que se pueden aplicar a la variable.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo de gráficos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# Variable objetivo
target = 'default.payment.next.month'

# Separar por tipo de variable
numericas = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
    'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

ordinales = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']

# ----------- Función para analizar variable numérica -----------
def analizar_numerica(df, var):
    print(f"\n📊 Variable numérica: {var}")
    display(df[var].describe())

    # Histograma
    sns.histplot(df[var], kde=True, bins=30)
    plt.title(f"Distribución de {var}")
    plt.show()

    # Boxplot
    sns.boxplot(x=df[var])
    plt.title(f"Outliers en {var}")
    plt.show()

    # Relación con la variable objetivo
    sns.boxplot(x=target, y=var, data=df)
    plt.title(f"{var} vs {target}")
    plt.show()

# ----------- Función para analizar variable categórica -----------
def analizar_categorica(df, var):
    print(f"\n🧩 Variable categórica: {var}")
    display(df[var].value_counts(normalize=True).round(3) * 100)

    # Gráfico de barras
    sns.countplot(x=var, data=df)
    plt.title(f"Distribución de {var}")
    plt.show()

    # Relación con la variable objetivo
    sns.barplot(x=var, y=target, data=df, ci=None)
    plt.title(f"Tasa de default por categoría de {var}")
    plt.show()
```

**Variables númericas**


```python
# ---------- Aplicar funciones a todas las variables ----------
print("📈 Análisis de variables individuales")

for var in numericas + ordinales:
    analizar_numerica(df, var)
```

    📈 Análisis de variables individuales
    
    📊 Variable numérica: LIMIT_BAL



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>167484.322667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>129747.661567</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>50000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>140000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>240000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1000000.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_2.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_3.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_4.png)
    


    
    📊 Variable numérica: AGE



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.485500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.217904</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_7.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_8.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_9.png)
    


    
    📊 Variable numérica: BILL_AMT1



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BILL_AMT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51223.330900</td>
    </tr>
    <tr>
      <th>std</th>
      <td>73635.860576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-165580.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3558.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22381.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>67091.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>964511.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_12.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_13.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_14.png)
    


    
    📊 Variable numérica: BILL_AMT2



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BILL_AMT2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>49179.075167</td>
    </tr>
    <tr>
      <th>std</th>
      <td>71173.768783</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-69777.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2984.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21200.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>64006.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>983931.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_17.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_18.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_19.png)
    


    
    📊 Variable numérica: BILL_AMT3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BILL_AMT3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.701315e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.934939e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.572640e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.666250e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.008850e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.016475e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.664089e+06</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_22.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_23.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_24.png)
    


    
    📊 Variable numérica: BILL_AMT4



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BILL_AMT4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>43262.948967</td>
    </tr>
    <tr>
      <th>std</th>
      <td>64332.856134</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-170000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2326.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19052.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>54506.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891586.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_27.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_28.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_29.png)
    


    
    📊 Variable numérica: BILL_AMT5



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BILL_AMT5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40311.400967</td>
    </tr>
    <tr>
      <th>std</th>
      <td>60797.155770</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-81334.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1763.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18104.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>50190.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>927171.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_32.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_33.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_34.png)
    


    
    📊 Variable numérica: BILL_AMT6



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BILL_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38871.760400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>59554.107537</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-339603.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1256.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17071.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49198.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>961664.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_37.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_38.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_39.png)
    


    
    📊 Variable numérica: PAY_AMT1



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_AMT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5663.580500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16563.280354</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2100.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5006.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>873552.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_42.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_43.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_44.png)
    


    
    📊 Variable numérica: PAY_AMT2



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_AMT2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.921163e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.304087e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.330000e+02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.009000e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.684259e+06</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_47.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_48.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_49.png)
    


    
    📊 Variable numérica: PAY_AMT3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_AMT3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5225.68150</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17606.96147</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>390.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1800.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4505.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>896040.00000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_52.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_53.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_54.png)
    


    
    📊 Variable numérica: PAY_AMT4



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_AMT4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4826.076867</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15666.159744</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>296.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4013.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>621000.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_57.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_58.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_59.png)
    


    
    📊 Variable numérica: PAY_AMT5



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_AMT5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4799.387633</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15278.305679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>252.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4031.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>426529.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_62.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_63.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_64.png)
    


    
    📊 Variable numérica: PAY_AMT6



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5215.502567</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17777.465775</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>117.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>528666.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_67.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_68.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_69.png)
    


    
    📊 Variable numérica: PAY_0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.016700</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.123802</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_72.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_73.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_74.png)
    


    
    📊 Variable numérica: PAY_2



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.133767</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.197186</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_77.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_78.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_79.png)
    


    
    📊 Variable numérica: PAY_3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.166200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.196868</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_82.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_83.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_84.png)
    


    
    📊 Variable numérica: PAY_4



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.220667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.169139</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_87.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_88.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_89.png)
    


    
    📊 Variable numérica: PAY_5



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.266200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.133187</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_92.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_93.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_94.png)
    


    
    📊 Variable numérica: PAY_6



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.291100</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.149988</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_97.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_98.png)
    



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_35_99.png)
    


**Categórica**


```python
for var in categoricas:
    analizar_categorica(df, var)
```

    
    🧩 Variable categórica: SEX



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>SEX</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>60.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39.6</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_37_2.png)
    


    /tmp/ipython-input-25-3392817890.py:57: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.
    
      sns.barplot(x=var, y=target, data=df, ci=None)



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_37_4.png)
    


    
    🧩 Variable categórica: EDUCATION



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>EDUCATION</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>46.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_37_7.png)
    


    /tmp/ipython-input-25-3392817890.py:57: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.
    
      sns.barplot(x=var, y=target, data=df, ci=None)



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_37_9.png)
    


    
    🧩 Variable categórica: MARRIAGE



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>MARRIAGE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>53.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_37_12.png)
    


    /tmp/ipython-input-25-3392817890.py:57: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.
    
      sns.barplot(x=var, y=target, data=df, ci=None)



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_37_14.png)
    


## Ranking de variables

En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo. Se utilizan técnicas como la correlación, el análisis de componentes principales (PCA) o la importancia de las variables en un modelo de aprendizaje automático.


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Elimina columnas no predictoras
X = df.drop(columns=["ID", "default.payment.next.month"])

# Estandarizar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
pca.fit(X_scaled)

# Graficar varianza explicada acumulada
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Número de componentes principales")
plt.ylabel("Varianza explicada acumulada")
plt.title("🔍 Análisis de Componentes Principales (PCA)")
plt.grid(True)
plt.show()

```

    /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128269 (\N{LEFT-POINTING MAGNIFYING GLASS}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_39_1.png)
    


**Se requieren alrededor de 12 componentes de PCA para explicar el 90% de la varianza**

Esto singifica que ¡:
- Alta dimensionalidad útil: Muchas variables aportan información relevante, por lo que eliminar variables sin análisis previo podría afectar el desempeño del modelo.

- No hay una fuerte redundancia: Si fueran muy redundantes, con 2 o 3 componentes podrías explicar casi toda la varianza.


```python
# Crear un DataFrame con los pesos (cargas) de cada variable en los componentes
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=X.columns
)

# Ver las variables más influyentes en el primer componente
top_PC1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
print("🔝 Variables más influyentes en el primer componente:")
print(top_PC1)

```

    🔝 Variables más influyentes en el primer componente:
    BILL_AMT4    0.353883
    BILL_AMT5    0.351752
    BILL_AMT3    0.349777
    BILL_AMT2    0.345797
    BILL_AMT6    0.344514
    BILL_AMT1    0.334385
    PAY_5        0.210960
    PAY_4        0.207038
    PAY_6        0.206412
    PAY_3        0.200213
    Name: PC1, dtype: float64


## Relación entre variables explicativas y variable objetivo

En esta sección se presenta un análisis de la relación entre las variables explicativas y la variable objetivo. Se utilizan gráficos como la matriz de correlación y el diagrama de dispersión para entender mejor la relación entre las variables. Además, se pueden utilizar técnicas como la regresión lineal para modelar la relación entre las variables.

**Correlación**


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular correlaciones con la variable objetivo
corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix["default.payment.next.month"].drop("default.payment.next.month").sort_values(key=abs, ascending=False)

# Mostrar top 10 variables más correlacionadas
plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.values[:10], y=target_corr.index[:10], palette="coolwarm")
plt.title("Top 10 Correlaciones con la variable objetivo")
plt.xlabel("Coeficiente de correlación")
plt.tight_layout()
plt.show()

```

    /tmp/ipython-input-19-1233923278.py:10: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x=target_corr.values[:10], y=target_corr.index[:10], palette="coolwarm")



    
![png](Entrega_1_data_summary_files/Entrega_1_data_summary_44_1.png)
    


**Rgresión múltiple**


```python
import statsmodels.api as sm

# 1. Definir X (variables explicativas) y y (variable objetivo)
X = df.drop(columns=["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]

# 2. Agregar constante (intercepto)
X = sm.add_constant(X)

# 3. Ajustar modelo
model = sm.OLS(y, X).fit()

# 4. Mostrar resumen
print(model.summary())

```

                                    OLS Regression Results                                
    ======================================================================================
    Dep. Variable:     default.payment.next.month   R-squared:                       0.124
    Model:                                    OLS   Adj. R-squared:                  0.123
    Method:                         Least Squares   F-statistic:                     184.5
    Date:                        Thu, 10 Jul 2025   Prob (F-statistic):               0.00
    Time:                                18:43:44   Log-Likelihood:                -14202.
    No. Observations:                       30000   AIC:                         2.845e+04
    Df Residuals:                           29976   BIC:                         2.865e+04
    Df Model:                                  23                                         
    Covariance Type:                    nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.3142      0.018     17.541      0.000       0.279       0.349
    LIMIT_BAL  -9.053e-08   2.16e-08     -4.193      0.000   -1.33e-07   -4.82e-08
    SEX           -0.0145      0.005     -3.130      0.002      -0.024      -0.005
    EDUCATION     -0.0151      0.003     -5.022      0.000      -0.021      -0.009
    MARRIAGE      -0.0238      0.005     -4.996      0.000      -0.033      -0.014
    AGE            0.0014      0.000      5.128      0.000       0.001       0.002
    PAY_0          0.0957      0.003     34.596      0.000       0.090       0.101
    PAY_2          0.0195      0.003      5.828      0.000       0.013       0.026
    PAY_3          0.0117      0.004      3.256      0.001       0.005       0.019
    PAY_4          0.0034      0.004      0.846      0.398      -0.004       0.011
    PAY_5          0.0057      0.004      1.324      0.185      -0.003       0.014
    PAY_6          0.0008      0.004      0.225      0.822      -0.006       0.008
    BILL_AMT1  -6.225e-07   1.14e-07     -5.453      0.000   -8.46e-07   -3.99e-07
    BILL_AMT2   1.587e-07    1.6e-07      0.990      0.322   -1.56e-07    4.73e-07
    BILL_AMT3   3.005e-08   1.51e-07      0.199      0.842   -2.66e-07    3.26e-07
    BILL_AMT4  -6.793e-08   1.57e-07     -0.432      0.666   -3.76e-07     2.4e-07
    BILL_AMT5  -2.049e-08   1.85e-07     -0.111      0.912   -3.82e-07    3.41e-07
    BILL_AMT6   1.153e-07   1.46e-07      0.789      0.430   -1.71e-07    4.02e-07
    PAY_AMT1   -7.437e-07   1.77e-07     -4.201      0.000   -1.09e-06   -3.97e-07
    PAY_AMT2   -2.092e-07   1.46e-07     -1.436      0.151   -4.95e-07    7.63e-08
    PAY_AMT3   -2.874e-08   1.69e-07     -0.170      0.865    -3.6e-07    3.02e-07
    PAY_AMT4   -2.521e-07   1.84e-07     -1.371      0.170   -6.13e-07    1.08e-07
    PAY_AMT5    -3.41e-07   1.91e-07     -1.787      0.074   -7.15e-07     3.3e-08
    PAY_AMT6    -9.77e-08   1.37e-07     -0.716      0.474   -3.65e-07     1.7e-07
    ==============================================================================
    Omnibus:                     4682.286   Durbin-Watson:                   2.013
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7285.821
    Skew:                           1.204   Prob(JB):                         0.00
    Kurtosis:                       3.178   Cond. No.                     2.10e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.1e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.


Las variables significativas en el modelo (que tienen un p valor < 0.05) son:

LIMIT_BAL  
SEX  
EDUCATION  
MARRIAGE  
AGE  
PAY_0  
PAY_2  
PAY_3  
BILL_AMT1  
PAY_AMT1  



```python
import os

for archivo in os.listdir('/content'):
    print(os.path.join('/content', archivo))

```

    /content/.config
    /content/drive
    /content/sample_data



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
!jupyter nbconvert --to markdown /content/drive/MyDrive/Colab_Notebooks/Entrega_1_data_summary.ipynb
```

    [NbConvertApp] WARNING | pattern '/content/drive/MyDrive/Colab_Notebooks/Entrega_1_data_summary.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    



```python
!jupyter nbconvert --to markdown /content/drive/Entrega_1_data_summary.ipynb
```

    [NbConvertApp] WARNING | pattern '/content/drive/Entrega_1_data_summary.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

