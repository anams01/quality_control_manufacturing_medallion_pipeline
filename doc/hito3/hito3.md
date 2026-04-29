<div class="portada">

**Asignatura:** Desarrollo y despliegue de soluciones Big Data  
**Máster:** Big Data y Análisis de Datos  
**Estudiante:** Ana Martín Serrano  
**Fecha de entrega:** Mayo 2026

</div>

<div class="page-break"></div>
---

## 4.1. Cuadernos y scripts de la fase de modelado

Esta sección describe la tercera fase del ciclo de vida del proyecto. A diferencia de las fases anteriores —cuyo código reside en el pipeline de DLT bajo el directorio `src/`— los cuadernos de esta fase se ejecutan de forma independiente o como tareas de un trabajo de Databricks, ubicados en la carpeta `notebooks` del repositorio. 

Todo el proceso está instrumentado con **MLflow** y gobernado por **Unity Catalog**, garantizando la trazabilidad de cada decisión y la reproducibilidad de cualquier resultado.

### Estructura de archivos

| Cuaderno / Script | Responsabilidad |
| :--- | :--- |
| `05_Training_Dataset_Generation.ipynb` | Genera el dataset de entrenamiento combinando la tabla spine con las tablas de características mediante *point-in-time join* del Feature Store. |
| `07_Utils.py` | Utilidades compartidas: configuración, partición temporal, clasificación de columnas y funciones de evaluación. |
| `07_Training_Job.ipynb` | Cuaderno de entrenamiento aislado. Recibe hiperparámetros, construye el pipeline y devuelve el modelo serializado. |
| `07_Evaluation_Job.ipynb` | Cuaderno de evaluación aislado. Carga el modelo desde su URI y devuelve las métricas y artefactos de diagnóstico. |
| `07_MLflow_Experimentation.ipynb` | Orquestador del *grid search*. Lanza las tareas de entrenamiento y evaluación en sesiones aisladas de Spark Connect. |
| `08_Utils.py` | Utilidades para el ciclo de producción: gestión de aliases y decisión de promoción. |
| `08_Production.ipynb` | Implementa el patrón *champion-challenger* para decidir la promoción del modelo a producción. |

> **Nota arquitectónica:** El aislamiento en cuadernos independientes mediante `dbutils.notebook.run()` garantiza sesiones de Spark Connect limpias. Esto previene el error `ML_CACHE_SIZE_OVERFLOW_EXCEPTION` común en entornos serverless cuando se acumulan modelos en la caché de la sesión.

---

## 4.2. Generación del conjunto de datos de entrenamiento

Se materializa el dataset a partir de las tablas de la capa oro y se persiste como tabla Delta estática: `gold_inspection_training_dataset`.

### 4.2.1. Cruce point-in-time con el Feature Store
El cruce se realiza mediante `fe.create_training_set` usando la tabla `gold_inspection_spine`. El parámetro `timestamp_lookup_key` es el mecanismo central para garantizar la **correctitud temporal**, recuperando la versión de las características válida en el instante exacto de la inspección y eliminando el *data leakage*.

### 4.2.2. Comprobaciones de calidad y resultados

| Métrica | Valor | Observación |
| :--- | :--- | :--- |
| Total de filas (spine) | 30.000.032 | Coincide con silver_inspections_labeled |
| Inspecciones válidas | ~29.865.000 | Usadas para entrenamiento |
| Clase 0 — Válidas | ~96,1% | Fuerte desbalance de clases |
| Clase 1 — Defectuosas | ~3,9% | Tasa con tendencia creciente (drift) |

---

## 4.3. Arquitectura del pipeline de preprocesado

El pipeline utiliza operaciones nativas de Spark para asegurar compatibilidad total con Spark Connect:

1.  **Imputación numérica:** Cálculo de la mediana para rellenar nulos.
2.  **Indexación categórica:** Conversión de strings a índices (con manejo de valores nuevos).
3.  **Codificación one-hot:** Expansión a vectores binarios.
4.  **Ensamblaje de vector:** Concatenación de características en un solo vector.
5.  **Escalado estándar:** Normalización vía `StandardScaler`.
6.  **Clasificador (LR):** Regresión logística binaria con pesos de clase dinámicos.

---

## 4.4. Optimización de hiperparámetros

### 4.4.1. Estrategia de partición temporal
| Partición | Período | Uso |
| :--- | :--- | :--- |
| Entrenamiento (70%) | Ene 2023 → Dic 2024 | Ajuste del modelo |
| Validación (15%) | Ene 2025 → Mar 2025 | Selección de hiperparámetros |
| Prueba (15%) | Abr 2025 → Jun 2025 | Evaluación final imparcial |

### 4.4.2. Resultados del grid search (AUC-PR)
| reg_param | elastic_net | AUC-PR val. | AUC-ROC val. | Umbral óptimo |
| :--- | :--- | :--- | :--- | :--- |
| **0.001** | **0.0** | **0.758** | **0.951** | **0.91** |
| 0.001 | 0.5 | 0.756 | 0.950 | 0.91 |
| 0.01 | 0.0 | 0.750 | 0.949 | 0.90 |

**Configuración ganadora:** `reg_param = 0.001, elastic_net_param = 0.0`.

---

## 4.5. Evaluación y Promoción (Champion-Challenger)

| Métrica | Conjunto prueba | Objetivo negocio | ¿Superado? |
| :--- | :--- | :--- | :--- |
| **Recall** | **0.78** | **≥ 75%** | **Sí** |
| **Falsos positivos** | **3.7%** | **< 4%** | **Sí** |

**Decisión:** El candidato se promueve a **champion** (versión 2 en Unity Catalog).

---

## 4.6. Consideraciones finales y ROI

* **Recall = 78%:** Mejora de 18 puntos sobre el sistema actual.
* **Ahorro proyectado:** El ahorro total estimado es de **~607.400 €/mes**.
* **Interpretabilidad:** El uso de Regresión Logística permite a los operarios auditar las predicciones mediante los coeficientes del modelo.