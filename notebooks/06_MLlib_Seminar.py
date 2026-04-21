# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Seminario de `MLlib`
# MAGIC
# MAGIC **Autor**: Juan Carlos Alfaro Jiménez
# MAGIC
# MAGIC Esta libreta es un seminario **autocontenido** de `Apache Spark MLlib` orientado a la detección de fraude en tarjetas de crédito. A diferencia del flujo principal de la asignatura (que integra el `Feature Store` de `Databricks` y `MLflow`), esta libreta opera directamente sobre la tabla `Delta` de entrenamiento que ya existe en `Unity Catalog` y cubre de extremo a extremo el ciclo de `MLlib`: desde la carga de datos hasta la evaluación de un modelo.
# MAGIC
# MAGIC El objetivo no es entrenar el mejor modelo posible, sino **mostrar cómo se trabaja con `MLlib`**: sus abstracciones fundamentales (`Transformer`, `Estimator`, `Pipeline`), su catálogo de transformadores de características, sus algoritmos de clasificación y su infraestructura de ajuste de hiperparámetros. En todos los casos se incluyen enlaces a la documentación oficial para que el alumno pueda profundizar por su cuenta en las técnicas que resulten más relevantes para su caso de uso.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Configuración e importaciones
# MAGIC
# MAGIC En esta sección se importan todos los componentes de `MLlib` que se utilizarán a lo largo del seminario y se definen los identificadores de las tablas siguiendo la convención de `Unity Catalog` (`catálogo.esquema.tabla`).

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.feature import (
    Binarizer,
    Bucketizer,
    ChiSqSelector,
    Imputer,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    QuantileDiscretizer,
    SQLTransformer,
    StandardScaler,
    StringIndexer,
    UnivariateFeatureSelector,
    VarianceThresholdSelector,
    VectorAssembler
)
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, StringType
import pandas as pd

# COMMAND ----------

catalog = "workspace"
database = "credit_card_fraud"

training_table = f"{catalog}.{database}.gold_fraud_training_dataset"

label_column = "is_fraud"
raw_prediction_column = "rawPrediction"
probability_column = "probability"
prediction_column = "prediction"

seed = 45127  # Random seed (fix it for reproducibility)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Carga del conjunto de datos desde `Delta`
# MAGIC
# MAGIC La primera diferencia visible con respecto a `scikit-learn` es que aquí los datos nunca «salen» del clúster hacia el nodo *driver*: `spark.table(...)` devuelve un `DataFrame` que representa un plan de ejecución distribuido, no una tabla en memoria. Todas las transformaciones posteriores se encolan como instrucciones dentro de ese plan y se materializan en los *workers* cuando se llama a una acción (`.count()`, `.collect()`, `.write`, etc.).
# MAGIC
# MAGIC La tabla `gold_fraud_training_dataset` ya tiene los datos limpios (sin registros con etiqueta nula) tal y como los dejó la libreta de generación del conjunto de entrenamiento. Solo necesitamos leerlos.
# MAGIC

# COMMAND ----------

df_raw = spark.table(training_table)

print(f"Rows: {df_raw.count():,}")
print(f"Columns: {len(df_raw.columns)}")
df_raw.printSchema()

# COMMAND ----------

df_raw.limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Conceptos fundamentales: `Transformer`, `Estimator` y `Pipeline`
# MAGIC
# MAGIC Antes de escribir una sola línea de preprocesado conviene entender las tres abstracciones sobre las que se construye todo `MLlib`. La confusión con `scikit-learn` es habitual porque los nombres son similares pero el comportamiento difiere.
# MAGIC
# MAGIC ### `Transformer`
# MAGIC
# MAGIC Un `Transformer` es un objeto que recibe un `DataFrame` y devuelve otro `DataFrame` con una o más columnas añadidas. **No aprende nada del dato**: siempre aplica la misma transformación determinista. El método de referencia es `.transform(...)`.
# MAGIC
# MAGIC **Ejemplos**: `VectorAssembler`, `SQLTransformer`, `Binarizer`, `Normalizer`, etc.
# MAGIC
# MAGIC ### `Estimator`
# MAGIC
# MAGIC Un `Estimator` sí aprende del dato. Primero se le llama `.fit(...)`, que consume el `DataFrame` completo para calcular los parámetros internos. El resultado de `.fit()` es un `Model`, que a su vez es un `Transformer` y expone el método `.transform()`.
# MAGIC
# MAGIC En el caso de los algoritmos de clasificación, `.transform()` es el equivalente al `.predict(...)` de `scikit-learn`: aplica los parámetros aprendidos durante el entrenamiento para añadir al `DataFrame` las columnas `rawPrediction`, `probability` y `prediction`.
# MAGIC
# MAGIC **Ejemplos**: `StandardScaler` (el `.fit()` calcula la media y la varianza; el modelo resultante aplica la normalización), `StringIndexer` (el `.fit()` aprende el vocabulario de categorías; el modelo resultante asigna un índice a cada una), `Imputer` (el `.fit()` calcula la media o mediana de cada columna; el modelo resultante sustituye los nulos por ese valor), cualquier algoritmo de clasificación como `LogisticRegression` (el `.fit()` aprende los coeficientes y el término independiente; el modelo resultante los aplica para predecir la probabilidad de cada clase).
# MAGIC
# MAGIC ### `Pipeline`
# MAGIC
# MAGIC Un `Pipeline` encadena una secuencia ordenada de `Transformer` y `Estimator`. Cuando se llama a `.fit()`, el `Pipeline` ejecuta los pasos en orden: cada `Estimator` llama a `.fit()` sobre la salida del paso anterior y genera un `Model`; cada `Transformer` llama directamente a `.transform()`. El resultado es un `PipelineModel`, que encapsula todos los modelos ajustados en secuencia y puede aplicarse con un único `.transform()` sobre nuevos datos.
# MAGIC
# MAGIC La ventaja frente a aplicar las transformaciones a mano es garantizar que las estadísticas aprendidas durante el entrenamiento (escalas, vocabularios, cuantiles, etc.) se aplican exactamente igual en validación, prueba e inferencia, evitando la **fuga de datos** (*data leakage*).
# MAGIC
# MAGIC > **Documentación oficial**: [`ML Pipelines`](https://spark.apache.org/docs/latest/ml-pipeline.html)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Preprocesado de características
# MAGIC
# MAGIC Esta sección muestra las técnicas de preprocesado de `MLlib` una a una, con su justificación teórica y su aplicación al contexto de detección de fraude. Todas ellas quedarán integradas en el `Pipeline` de la sección siguiente.
# MAGIC
# MAGIC Para entender cómo se encadenan estas transformaciones, es fundamental comprender una regla de diseño arquitectónico clave en `MLlib`. El preprocesado se divide siempre en dos fases o "mundos", separados por el componente `VectorAssembler`:
# MAGIC
# MAGIC * **Fase tabular (`inputCols` e `inputCol`)**: Las transformaciones iniciales operan sobre columnas de forma independiente (por ejemplo, imputar valores perdidos en la edad o codificar el país). Como la operación transforma sin mirar el resto de características, la `API` de estos transformadores utiliza los parámetros `inputCol` o `inputCols`.
# MAGIC * **Fase vectorial (`featuresCol` e `inputCol`)**: A partir del `VectorAssembler`, las características viajan empaquetadas en un único vector matemático. Sin embargo, notarás una diferencia en los parámetros según la interfaz del componente:
# MAGIC     * **Transformaciones de datos (`inputCol`)**: Los componentes que se limitan a aplicar una transformación matemática de entrada y salida (como escalar el vector con `StandardScaler`) utilizan la interfaz genérica `inputCol` y `outputCol`.
# MAGIC     * **Algoritmos predictivos y estadísticos (`featuresCol`)**: Los modelos de clasificación y los selectores de características necesitan evaluar la matriz de variables (`X`) frente a la etiqueta objetivo (`y`). Para reflejar esta relación matemática, la `API` exige identificar estos campos explícitamente mediante `featuresCol` y `labelCol`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Antes de definir ningún transformador, clasificamos las columnas del esquema en tres grupos según su tipo y su semántica. Esta clasificación determina qué tratamiento recibirá cada columna a lo largo del preprocesado:
# MAGIC
# MAGIC * **Numéricas**: columnas de tipo entero o decimal que representan magnitudes reales (`amount`, `age`, agregaciones de ventana, etc.).
# MAGIC * **Booleanas**: columnas almacenadas como entero (`0` o `1`) que representan *flags* binarios (`two_fa_enabled`, `cross_border`, etc.).
# MAGIC * **Categóricas**: columnas de tipo `String`.
# MAGIC
# MAGIC Las siguientes columnas se excluyen explícitamente del vector de características antes de la clasificación:
# MAGIC
# MAGIC * **`is_fraud`**: la etiqueta de supervisión. Incluirla en las características introduciría una fuga de datos directa que haría que el modelo aprendiera a copiar la respuesta en lugar de generalizarla.
# MAGIC * **`customer_id`**, **`transaction_id`**: identificadores de fila sin información predictiva. Su cardinalidad es igual al número de filas, por lo que el modelo solo memorizaría ejemplos de entrenamiento.
# MAGIC * **`merchant_id`**: identificador de alta cardinalidad. Con cientos de miles de valores únicos, el `OneHotEncoder` generaría un vector de dimensión inviable. La información del comercio ya está capturada por `mcc_category` y `merchant_country`.
# MAGIC * **`mcc_code`**: redundante con `mcc_category`, que es su versión legible y de menor cardinalidad.
# MAGIC * **`timestamp`**: columna de partición temporal. No es una característica predictiva sino un metadato operativo.
# MAGIC

# COMMAND ----------

numeric_types = {"IntegerType", "LongType", "FloatType", "DoubleType", "DecimalType"}
boolean_types = {"BooleanType"}
categorical_types = {"StringType"}

# Columns stored as integers but representing binary flags, not magnitudes
binary_flag_columns = [
    "cross_border", 
    "is_tor_or_vpn",
    "ip_country_match",
    "device_fingerprint_known",
    "two_fa_enabled",
    "email_verified",
    "phone_verified"
]

exclude_columns = [
    label_column,
    "customer_id",
    "transaction_id",
    "merchant_id",
    "mcc_code",
    "timestamp"
]

numeric_columns = []
boolean_columns = []
categorical_columns = []

for field in df_raw.schema.fields:
    column_name = field.name
    type_name = type(field.dataType).__name__

    if column_name in exclude_columns:
        continue
    if column_name in binary_flag_columns:
        boolean_columns.append(column_name)
    elif type_name in numeric_types:
        numeric_columns.append(column_name)
    elif type_name in boolean_types:
        boolean_columns.append(column_name)
    elif type_name in categorical_types:
        categorical_columns.append(column_name)

print(f"Numeric columns: {numeric_columns}")
print(f"Boolean columns: {boolean_columns}")
print(f"Categorical columns: {categorical_columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.1. Imputación de valores nulos (`Imputer`)
# MAGIC
# MAGIC No todos los nulos en este conjunto de datos tienen el mismo significado, y la estrategia de imputación debe reflejarlo. Hay tres casos distintos:
# MAGIC
# MAGIC * **Columnas de perfil** (`age`, `city_tier`, `num_cards_issued`, `loyalty_points_balance`): los nulos son consecuencia del `PiT` *join*. Muchos clientes no tienen un registro de perfil válido en la fecha de la transacción, por lo que el valor es desconocido. Se imputa con la **mediana**, más robusta ante *outliers* que la media.
# MAGIC
# MAGIC * **Columnas de agregación estadística** (`avg_*`, `max_amount_*`, `min_amount_*`): los nulos son **intencionados** en el *pipeline* de construcción de características. La media o el máximo de una ventana vacía es matemáticamente indefinido, y imputar `0` sería factualmente incorrecto (un importe medio de `0` no es lo mismo que no haber tenido actividad). Se imputa también con la **mediana**.
# MAGIC
# MAGIC * **Columnas de agregación de conteo y suma** (`count_*`, `sum_*`, `distinct_*`, etc.): el *pipeline* ya las inicializa a `0` cuando la ventana está vacía, por lo que llegan sin nulos y no requieren imputación.
# MAGIC
# MAGIC El `Imputer` de `MLlib` es un `Estimator` que aprende la estadística de imputación durante `.fit()` y la aplica durante `.transform()`. **Solo funciona con columnas numéricas**; las categóricas con nulos se tratan en el paso de `StringIndexer` mediante el parámetro `handleInvalid`.
# MAGIC
# MAGIC > **Documentación oficial**: [`Imputer`](https://spark.apache.org/docs/latest/ml-features.html#imputer)

# COMMAND ----------

agg_zero_prefixes = ("count_", "sum_", "count_cross_border_", "count_tor_vpn_", "count_3ds_failed_", "num_fraud_", "spend_", "distinct_")
agg_null_prefixes = ("avg_", "max_amount", "min_amount")

agg_zero_columns = [column for column in numeric_columns if column.startswith(agg_zero_prefixes)]
agg_null_columns = [column for column in numeric_columns if column.startswith(agg_null_prefixes)]
profile_num_columns = [
    column for column in numeric_columns
    if not column.startswith(agg_zero_prefixes) and not column.startswith(agg_null_prefixes)
]

imputer_input_columns = profile_num_columns + agg_null_columns
imputer_output_columns = [f"{column}_imp" for column in imputer_input_columns]
imputer_strategy = "median"

print(f"Profile numeric (median imputation): {profile_num_columns}")
print(f"Aggregation numeric (median imputation): {agg_null_columns}")
print(f"Aggregation numeric (already 0, no imputation): {agg_zero_columns}")

# Single imputer for both profile and intentionally-null aggregation columns.
# Both share the same strategy (median), so they can be merged into one stage.
imputer = Imputer(
    inputCols = imputer_input_columns,
    outputCols = imputer_output_columns,
    strategy = imputer_strategy
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.2. Conversión de variables booleanas (`SQLTransformer`)
# MAGIC
# MAGIC Las columnas booleanas de nuestro conjunto de datos están almacenadas como enteros (`0` o `1`) en lugar de `BooleanType`, por lo que técnicamente `MLlib` podría incluirlas en el vector numérico sin problema. Sin embargo, las tratamos de forma separada por semántica: son *flags* binarios, no magnitudes, y no tiene sentido escalarlos ni imputarlos con la mediana.
# MAGIC
# MAGIC Este paso combina en una sola expresión `SQL` **dos operaciones inseparables conceptualmente**: el cast a `DOUBLE` y la imputación a `0.0`. Columnas como `two_fa_enabled`, `email_verified` y `phone_verified` pueden llegar con `null` cuando el `PiT` join no encontró registro de perfil del cliente en la fecha de la transacción. La imputación correcta para un flag de seguridad ausente es siempre `0.0`: la ausencia de constancia de que esté activado es funcionalmente equivalente a no estarlo.
# MAGIC
# MAGIC Esta imputación **no se delega al `Imputer`** por dos razones. Primero, el `Imputer` opera antes de que las columnas `_dbl` existan en el `DataFrame`. Segundo, el `Imputer` está diseñado para magnitudes continuas donde la mediana tiene sentido estadístico; para un *flag* binario, imputar con `COALESCE(..., 0.0)` directamente en el *cast* es semánticamente más preciso y estructuralmente más limpio.
# MAGIC
# MAGIC > **Documentación oficial**: [`SQLTransformer`](https://spark.apache.org/docs/latest/ml-features.html#sqltransformer)
# MAGIC

# COMMAND ----------

# Generate a single SQL statement that casts all boolean columns in one pass,
# replacing nulls with 0.0.
# __THIS__ is the SQLTransformer reserved keyword that refers to the input DataFrame.
boolean_cast_expressions = ", ".join([
    f"COALESCE(CAST({column} AS DOUBLE), 0.0) AS {column}_dbl"
    for column in boolean_columns
])

# Output column names follow the _dbl suffix convention to distinguish them
# from the original integer columns, which are kept in the DataFrame unchanged.
boolean_output_columns = [f"{column}_dbl" for column in boolean_columns]

boolean_statement = f"SELECT *, {boolean_cast_expressions} FROM __THIS__"
boolean_transformer = SQLTransformer(statement = boolean_statement)

print("Boolean columns cast to double:", boolean_output_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.3. Ingeniería de nuevas características (`SQLTransformer`)
# MAGIC
# MAGIC La *ingeniería de características* es el proceso de crear nuevas columnas a partir de las existentes que capturen información relevante de forma más explícita para el modelo. `SQLTransformer` permite expresar cualquier transformación como una sentencia `Spark SQL`, lo que lo hace extraordinariamente flexible.
# MAGIC
# MAGIC Antes de definir las transformaciones conviene hacer una distinción arquitectónica importante:
# MAGIC
# MAGIC * **Características que dependen del historial del cliente** (*ratios* entre ventanas temporales, velocidades de gasto, etc.) deberían calcularse en el *pipeline* de construcción de `Feature Store` (`gold_customer_aggregations`), no aquí. Ese es exactamente su propósito materializar señales comportamentales de forma que estén disponibles en tiempo real durante la inferencia. Añadirlas en este paso sería duplicar lógica que ya existe y que además está garantizada como libre de *data leakage* por el `PiT` *join*.
# MAGIC
# MAGIC * **Características que dependen únicamente de la transacción actual** son las que sí corresponde crear aquí. La transacción llega con los siguientes campos en crudo: `amount`, `currency`, `mcc_category`, `cross_border`, `is_tor_or_vpn`, `ip_country_match`, `device_fingerprint_known`, `payment_method`, `device_type`, `three_ds_result` y `merchant_country`. Cualquier combinación de estos campos que aporte señal adicional al modelo es un candidato legítimo para este paso.
# MAGIC
# MAGIC | Nueva característica | Fórmula | Interpretación |
# MAGIC |---|---|---|
# MAGIC | `is_high_risk_method` | `CAST((is_tor_or_vpn = 1 OR three_ds_result = 'FAILED') AS INT)` | La transacción usa un canal anónimo o ha fallado la autenticación fuerte. |
# MAGIC | `is_foreign_ip` | `CAST((ip_country_match = 0) AS INT)` | La `IP` no corresponde al país del comercio. |
# MAGIC | `is_unrecognized_device` | `CAST((device_fingerprint_known = 0) AS INT)` | El dispositivo no ha sido visto antes para este cliente. |
# MAGIC | `amount_log` | `LOG(amount + 1)` | Suaviza la distribución asimétrica del importe de la transacción. |
# MAGIC | `is_cross_border_online` | `CAST((cross_border = 1 AND payment_method = 'online') AS INT)` | Compra internacional *online*: combinación de mayor riesgo. |
# MAGIC
# MAGIC > **Regla de oro**: toda característica creada aquí debe poder calcularse con la información disponible **en el momento de la inferencia**, es decir, únicamente con los campos de la transacción entrante. Cualquier característica que requiera el historial del cliente pertenece al `Feature Store`, no a este paso.
# MAGIC
# MAGIC > **Documentación oficial**: [`SQLTransformer`](https://spark.apache.org/docs/latest/ml-features.html#sqltransformer)

# COMMAND ----------

feature_engineering_statement = """
    SELECT *,
        -- Transaction uses an anonymous channel or strong authentication has failed
        CAST((is_tor_or_vpn = 1 OR three_ds_result = 'FAILED') AS INT) AS is_high_risk_method,

        -- IP does not match the merchant country
        CAST((ip_country_match = 0) AS INT) AS is_foreign_ip,

        -- Device has not been seen before for this customer
        CAST((device_fingerprint_known = 0) AS INT) AS is_unrecognized_device,

        -- Log transformation to smooth the skewed distribution of transaction amounts
        LOG(amount + 1) AS amount_log,

        -- Cross-border online transaction: highest risk combination
        CAST((cross_border = 1 AND payment_method = 'online') AS INT) AS is_cross_border_online
    FROM __THIS__
"""

engineered_columns = [
    "is_high_risk_method",
    "is_foreign_ip",
    "is_unrecognized_device",
    "amount_log",
    "is_cross_border_online"
]

feature_engineer = SQLTransformer(statement = feature_engineering_statement)

print("Engineered columns:", engineered_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.4. Codificación de variables categóricas (`StringIndexer` + `OneHotEncoder`)
# MAGIC
# MAGIC Los algoritmos de `MLlib` (como casi cualquier algoritmo de aprendizaje automático) operan sobre vectores numéricos. Las columnas de tipo cadena (`StringType`) deben convertirse a números antes de poder incluirse en el vector de características.
# MAGIC
# MAGIC `MLlib` proporciona una cadena canónica de dos pasos:
# MAGIC
# MAGIC 1. **`StringIndexer`**: convierte cada categoría textual en un índice numérico entero. Las categorías se ordenan por frecuencia descendente: la categoría más frecuente recibe el índice `0`, la siguiente el `1`, y así sucesivamente. Esta elección minimiza el número de colisiones en tablas de *hash* internas.
# MAGIC
# MAGIC El parámetro `handleInvalid` controla qué ocurre si durante el `.transform()` aparece una categoría que no estaba en el vocabulario aprendido durante el `.fit()`:
# MAGIC - `"error"` (por defecto): lanza una excepción.
# MAGIC - `"skip"`: descarta la fila completa. Peligroso si las filas representan transacciones de inferencia.
# MAGIC - `"keep"`: asigna un índice especial a la categoría desconocida. **Esta es la opción correcta para producción**: evita errores si aparece un nuevo país, tipo de tarjeta o categoría de comercio que no existía en el entrenamiento.
# MAGIC
# MAGIC 2. **`OneHotEncoder`**: convierte el índice numérico en un vector *one-hot* disperso (`SparseVector`). Por defecto descarta la última categoría (codificación `drop = "last"`) para evitar la multicolinealidad perfecta con los modelos lineales.
# MAGIC
# MAGIC > **Importante**: `OneHotEncoder` **no es** un `Estimator` que aprende vocabulario. Su función es únicamente convertir el índice entero a vector disperso. El aprendizaje del vocabulario ocurre en el `StringIndexer`. Por eso ambos pasos deben ir **siempre en este orden** dentro del `Pipeline`.
# MAGIC
# MAGIC > **Documentación oficial**: [`StringIndexer`](https://spark.apache.org/docs/latest/ml-features.html#stringindexer) y [`OneHotEncoder`](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)
# MAGIC

# COMMAND ----------

string_indexer_input_columns = categorical_columns
string_indexer_output_columns = [f"{column}_idx" for column in categorical_columns]  # Output columns receive the "_idx" suffix for easy identification
string_indexer_handle_invalid = "keep"  # In production always "keep", never "error"
string_indexer_order_type = "frequencyDesc"  # Most frequent category → index 0

string_indexer = StringIndexer(
    inputCols = string_indexer_input_columns,
    outputCols = string_indexer_output_columns,
    handleInvalid = string_indexer_handle_invalid,
    stringOrderType = string_indexer_order_type
)

ohe_input_columns = string_indexer_output_columns
ohe_output_columns = [f"{column}_ohe" for column in categorical_columns]
ohe_handle_invalid = "keep"
ohe_drop_last = True  # Avoids perfect multicollinearity in linear models

ohe = OneHotEncoder(
    inputCols = ohe_input_columns,
    outputCols = ohe_output_columns,
    handleInvalid = ohe_handle_invalid,
    dropLast = ohe_drop_last
)

print("Category columns to index:", string_indexer_input_columns)
print("Integer index columns:", string_indexer_output_columns)
print("One-hot encoded columns:", ohe_output_columns)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ### 4.5. Discretización (`Bucketizer` y `QuantileDiscretizer`)
# MAGIC
# MAGIC La discretización convierte una columna continua en una columna categórica de intervalos. Es útil cuando:
# MAGIC
# MAGIC * La relación con la etiqueta no es monótona (por ejemplo, el fraude ocurre más frecuentemente a importes muy bajos y muy altos, pero no a importes medios).
# MAGIC * Se quiere capturar umbrales de negocio conocidos (por ejemplo, transacciones por encima de 1000 euros son de alto riesgo).
# MAGIC
# MAGIC `Bucketizer` requiere que el usuario especifique explícitamente los límites de los intervalos. `QuantileDiscretizer` aprende automáticamente los límites a partir de los cuantiles del dato, por lo que es un `Estimator`.
# MAGIC
# MAGIC > **Documentación oficial**: [`Bucketizer`](https://spark.apache.org/docs/latest/ml-features.html#bucketizer) y [`QuantileDiscretizer`](https://spark.apache.org/docs/latest/ml-features.html#quantilediscretizer)
# MAGIC

# COMMAND ----------

# Example 1: Manual splits on the transaction amount.
# Four buckets: [0, 50), [50, 200), [200, 1000), [1000, ∞)
# Amounts above 1000 are considered high-value and warrant closer scrutiny.
bucketizer_input_column = "amount"
bucketizer_output_column = "amount_bucket"
bucketizer_splits = [0.0, 50.0, 200.0, 1000.0, float("inf")]
bucketizer_handle_invalid = "keep"  # Nulls are assigned to a special bucket

bucketizer = Bucketizer(
    inputCol = bucketizer_input_column,
    outputCol = bucketizer_output_column,
    splits = bucketizer_splits,
    handleInvalid = bucketizer_handle_invalid
)

# Example 2: Quantile splits on the number of transactions in the last 30 days.
# Learns the 4 quartile boundaries from the data during .fit() and discretizes accordingly.
quantile_input_column = "count_tx_30d"
quantile_output_column = "count_tx_30d_quantile"
quantile_num_buckets = 4
quantile_relative_error = 0.01  # Approximation precision (0 = exact but slow)
quantile_handle_invalid = "keep"

quantile_discretizer = QuantileDiscretizer(
    inputCol = quantile_input_column,
    outputCol = quantile_output_column,
    numBuckets = quantile_num_buckets,
    relativeError = quantile_relative_error,
    handleInvalid = quantile_handle_invalid
)

print(f"Manual splits on {bucketizer_input_column}: {bucketizer_splits}")
print(f"Quantile discretizer: {quantile_num_buckets} buckets on {quantile_input_column}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.6. Ensamblaje del vector de características (`VectorAssembler`)
# MAGIC
# MAGIC Ahora que ya tenemos todas las transformaciones previas **columna a columna** en formato tabular perfectamente definidas, ha llegado el momento de unificarlas. `VectorAssembler` actúa como el puente estructural entre el formato tabular tradicional de los `DataFrame` (columnas individuales) y el formato matemático que exigen los algoritmos de `MLlib`: un único objeto `Vector` (denso o disperso) por fila. Recibe una lista de columnas numéricas o de tipo vector y las concatena.
# MAGIC
# MAGIC A partir de este punto, **las transformaciones restantes (como el escalado o la selección estadística) y los modelos de clasificación operarán sobre este vector único en lugar de sobre columnas sueltas.**
# MAGIC
# MAGIC El vector final está compuesto por cinco grupos de columnas, cada uno procedente de una etapa anterior del `Pipeline`:
# MAGIC
# MAGIC * **Columnas numéricas imputadas** (`_imp`): las versiones imputadas de las columnas de perfil y de las agregaciones estadísticas. Las columnas originales sin el sufijo `_imp` **no se incluyen**: han sido reemplazadas por sus versiones imputadas y añadirlas supondría duplicar información con nulos.
# MAGIC * **Columnas de agregación a cero**: las columnas de conteo y suma que el *pipeline* del `Feature Store` ya inicializa a `0` cuando la ventana está vacía. No necesitan imputación y se incluyen directamente.
# MAGIC * **Columnas booleanas** (`_dbl`): los *flags* binarios casteados a `double`. Las columnas originales enteras no se incluyen para evitar duplicidades.
# MAGIC * **Columnas categóricas codificadas** (`_ohe`): los vectores dispersos generados por `OneHotEncoder`. Las columnas originales de texto y sus índices intermedios (`_idx`) no se incluyen: son pasos intermedios, no características finales.
# MAGIC * **Columnas de ingeniería de características**: las nuevas señales creadas a partir de los campos de la transacción actual (`is_high_risk_method`, `amount_log`, etc.).
# MAGIC
# MAGIC El parámetro `handleInvalid` es crítico para definir la robustez del flujo frente a valores nulos o `NaN` residuales:
# MAGIC
# MAGIC * **`"error"` (recomendado)**: lanza una excepción y detiene la ejecución. Es la opción más segura en producción porque obliga a garantizar que las etapas previas de imputación cubren el 100% de los casos.
# MAGIC * **`"skip"`**: descarta silenciosamente la fila completa. Es muy peligroso en inferencia, ya que provocaría la pérdida de predicciones para transacciones legítimas.
# MAGIC * **`"keep"`**: preserva la fila inyectando valores `NaN` en el vector resultante. Aunque actúa como red de seguridad para no perder la transacción, la mayoría de los algoritmos de modelado fallarán en el siguiente paso al intentar procesar ese `NaN`.
# MAGIC
# MAGIC > **Documentación oficial**: [`VectorAssembler`](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)

# COMMAND ----------

assembler_input_columns = (
    imputer_output_columns  # Numeric profile + null aggregation columns (median imputed)
    + agg_zero_columns  # Aggregation columns already at 0, no imputation needed
    + boolean_output_columns  # Boolean flags cast to double
    + ohe_output_columns  # Categorical columns encoded as one-hot encoded vectors
    + engineered_columns  # New features derived from the current transaction fields
)

assembler_handle_invalid = "error"  # Fails fast if any nulls slipped through our imputation logic
assembler_output_column = "features"

assembler = VectorAssembler(
    inputCols = assembler_input_columns,
    outputCol = assembler_output_column,
    handleInvalid = assembler_handle_invalid
)

print(f"Imputed numeric columns: {imputer_output_columns}")
print(f"Zero aggregation columns: {agg_zero_columns}")
print(f"Boolean columns: {boolean_output_columns}")
print(f"One-hot encoded categorical columns: {ohe_output_columns}")
print(f"Engineered columns: {engineered_columns}")
print(f"Total columns in assembler: {len(assembler_input_columns)}")
print(f"Note: One-hot encoded columns each expand into multiple dimensions. The actual vector size is determined after pipeline.fit().")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.7. Selección de características (`ChiSqSelector`, `VarianceThresholdSelector` y `UnivariateFeatureSelector`)
# MAGIC
# MAGIC El vector final aumenta considerablemente su dimensionalidad tras el *one-hot encoding* de variables categóricas como `country` u `occupation`. La selección de características reduce la dimensionalidad, elimina ruido y acelera el entrenamiento.
# MAGIC
# MAGIC `MLlib` ofrece varios selectores:
# MAGIC
# MAGIC | Selector | Criterio | Tipo |
# MAGIC |---|---|---|
# MAGIC | `ChiSqSelector` | Prueba `χ²` entre cada característica y la etiqueta | `Estimator` (estadístico) |
# MAGIC | `VarianceThresholdSelector` | Elimina columnas con varianza menor al umbral | `Estimator` (estadístico) |
# MAGIC | `UnivariateFeatureSelector` | ANOVA F-test, `χ²`, regresión | `Estimator` (estadístico) |
# MAGIC
# MAGIC Es importante tener en cuenta las restricciones de cada selector antes de incluirlo en un
# MAGIC `Pipeline`:
# MAGIC
# MAGIC * **`ChiSqSelector`**: exige que **todas** las características del vector sean discretas (categóricas). Si el vector contiene columnas continuas como `amount` o las agregaciones de ventana temporal, el selector lanzará una excepción en tiempo de ejecución. Para usarlo correctamente habría que discretizar previamente todas las columnas continuas con `Bucketizer` o `QuantileDiscretizer`.
# MAGIC * **`VarianceThresholdSelector`**: no tiene restricciones sobre el tipo de característica. Es la opción más segura para aplicar directamente sobre el vector mixto que producimos en este seminario.
# MAGIC * **`UnivariateFeatureSelector`**: permite declarar explícitamente qué tipo de datos contiene el vector (`featureType`) y la etiqueta (`labelType`). Sin embargo, **aplica el mismo test a todo el vector**. Al configurarlo como `continuous` frente a una etiqueta `categorical`, el algoritmo aplicará un test ANOVA a todas las columnas. Aunque el vector contenga variables categóricas codificadas (ceros y unos del *one-hot encoding*), las evaluará matemáticamente como números continuos, una aproximación pragmática y ampliamente aceptada en `Spark` para vectores mixtos.
# MAGIC
# MAGIC > **Documentación oficial**: [`ChiSqSelector`](https://spark.apache.org/docs/latest/ml-features.html#chisqselector), [`VarianceThresholdSelector`](https://spark.apache.org/docs/latest/ml-features.html#variancethresholdselector) y [`UnivariateFeatureSelector`](https://spark.apache.org/docs/latest/ml-features.html#univariatefeatureselect)

# COMMAND ----------

# Chi-square selector: selects the k best features by statistical association with the label.
# Excluded from the base pipeline: requires all features to be discrete.
# To use it correctly, all continuous columns would need to
# be discretized first before being passed to this selector.
chi_selector_input_column = assembler_output_column
chi_selector_output_column = "features_chi_selected"
chi_selector_type = "numTopFeatures"
chi_selector_top_features = 40

chi_selector = ChiSqSelector(
    featuresCol = chi_selector_input_column,
    outputCol = chi_selector_output_column,
    labelCol = label_column,
    selectorType = chi_selector_type,
    numTopFeatures = chi_selector_top_features
)

# Variance threshold selector: removes quasi-constant features (variance below threshold).
# Applied directly on the raw assembled vector, before scaling.
# Standarization sets all variances to 1.0, making this filter useless if applied after.
var_selector_input_column = assembler_output_column
var_selector_output_column = "features_var_filtered"
var_selector_threshold = 0.01

var_selector = VarianceThresholdSelector(
    featuresCol = var_selector_input_column,
    outputCol = var_selector_output_column,
    varianceThreshold = var_selector_threshold
)

# Univariate feature selector: applies the most appropriate statistical test based on
# the declared feature and label types. For continuous features with a binary label,
# it applies an ANOVA F-test. For categorical features with a binary label, it applies
# a chi-square test.
univariate_selector_input_column = assembler_output_column
univariate_selector_output_column = "features_univariate_selected"
univariate_feature_type = "continuous"
univariate_label_type = "categorical"
univariate_selection_mode = "numTopFeatures"
univariate_selector_num_features = 40

univariate_selector = (
    UnivariateFeatureSelector(
        featuresCol = univariate_selector_input_column,
        outputCol = univariate_selector_output_column,
        labelCol = label_column,
        selectionMode = univariate_selection_mode
    )
    .setFeatureType(univariate_feature_type)
    .setLabelType(univariate_label_type)
    .setSelectionThreshold(univariate_selector_num_features)
)

print(f"Chi-square selector: top {chi_selector_top_features} features (excluded from base pipeline)")
print(f"Variance threshold selector: variance > {var_selector_threshold}")
print(f"Univariate selector: top {univariate_selector_num_features} features (ANOVA F-test for continuous features)")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.8. Escalado de características (`StandardScaler`, `MinMaxScaler` y `Normalizer`)
# MAGIC
# MAGIC El escalado es crítico para los **modelos basados en gradiente** (regresión logística, máquinas vectores soporte, redes neuronales) y para los **algoritmos basados en distancia** (vecinos más cercanos, k-medias). Los árboles de decisión y los modelos ensemble basados en ellos (*random forests*, *gradient boosting*) son **invariantes al escalado**: no necesitan que las características estén en la misma escala porque sus decisiones se basan en umbrales sobre características individuales, no en distancias.
# MAGIC
# MAGIC `MLlib` ofrece tres estrategias principales:
# MAGIC
# MAGIC | Escalador | Transformación | Cuándo usarlo |
# MAGIC |---|---|---|
# MAGIC | `StandardScaler` | `(x - μ) / σ` | Media 0, desviación 1. Para modelos que asumen distribución Gaussiana o usan regularización. |
# MAGIC | `MinMaxScaler` | `(x - min) / (max - min)` | Rango [0, 1]. Cuando la distribución no es Gaussiana pero los límites son conocidos. |
# MAGIC | `Normalizer` | `x / ‖x‖_p` | Normaliza cada *fila* (no cada columna) a norma unitaria. Para modelos de texto o similitud. |
# MAGIC
# MAGIC Para nuestro caso de uso, `StandardScaler` es la elección más habitual al entrenar regresión logística con regularización.
# MAGIC
# MAGIC > **Documentación oficial**: [`StandardScaler`](https://spark.apache.org/docs/latest/ml-features.html#standardscaler), [`MinMaxScaler`](https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler) y [`Normalizer`](https://spark.apache.org/docs/latest/ml-features.html#normalizer)
# MAGIC

# COMMAND ----------

# Learns mean and variance during .fit() and applies them in .transform()
# withMean = False when the vector is sparse: centering a sparse
# vector produces a dense one, which can cause memory issues at scale.
scaler_input_column = var_selector_output_column
scaler_output_column = "features_scaled"
with_mean = False
with_std = True

standard_scaler = StandardScaler(
    inputCol = scaler_input_column,
    outputCol = scaler_output_column,
    withMean = with_mean,
    withStd = with_std
)

# Scales each feature to the [0, 1] range
minmax_input_column = var_selector_output_column
minmax_output_column = "features_minmax"
minmax_min = 0.0
minmax_max = 1.0

minmax_scaler = MinMaxScaler(
    inputCol = minmax_input_column,
    outputCol = minmax_output_column,
    min = minmax_min,
    max = minmax_max
)

# Normalizes each row to unit norm (L2 by default).
# Operates row-wise: it scales the entire feature
# vector of each sample so that its L2 norm equals 1.
normalizer_input_column = var_selector_output_column
normalizer_output_column = "features_normalized"
normalizer_p = 2.0

normalizer = Normalizer(
    inputCol = normalizer_input_column,
    outputCol = normalizer_output_column,
    p = normalizer_p
)

print(f"Standard scaler: zero mean = {with_mean}, unit std = {with_std}")
print(f"Min-max scaler: range = [{minmax_min}, {minmax_max}]")
print(f"Normalizer: L{int(normalizer_p)} norm")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Construcción del *pipeline* de preprocesado completo
# MAGIC
# MAGIC Un `Pipeline` bien diseñado encapsula **todas** las decisiones de preprocesado como una secuencia ordenada y reproducible. La clave es que **ninguna transformación se calcula fuera del `Pipeline`**: todo lo que se aprende del dato (medianas para imputar, vocabularios para indexar, cuantiles para discretizar, etc.) se aprende **solo sobre el conjunto de entrenamiento** y se aplica con los mismos parámetros sobre validación, prueba e inferencia.
# MAGIC
# MAGIC Si calculásemos, por ejemplo, la mediana de `age` sobre todo el `DataFrame` antes de hacer la división entrenamiento y prueba, estaríamos introduciendo información del conjunto de prueba en el proceso de ajuste y, por tanto, cometeríamos un *data leakage*. El `Pipeline` cierra esa vía por diseño.
# MAGIC
# MAGIC La secuencia de pasos que definimos aquí es:
# MAGIC
# MAGIC 1. **Imputación** (`Imputer`): imputa la mediana en las columnas de perfil numérico y en las agregaciones estadísticas con nulos intencionados (`avg_*`, `max_amount_*` y `min_amount_*`).
# MAGIC 2. **Conversión de booleanos** (`SQLTransformer`): castea los *flags* binarios a `double`.
# MAGIC 3. **Ingeniería de características** (`SQLTransformer`): crea nuevas señales a partir de los campos de la transacción actual.
# MAGIC 4. **Indexación** (`StringIndexer`): convierte cada categoría textual a un índice entero.
# MAGIC 5. **Codificación** (`OneHotEncoder`): convierte los índices a vectores dispersos.
# MAGIC 6. **Ensamblaje** (`VectorAssembler`): concatena todas las columnas en el vector `features`.
# MAGIC 7. **Selección de características** (`VarianceThresholdSelector`): elimina características cuasi-constantes que no aportan información discriminativa. `ChiSqSelector` queda excluido del *pipeline* base porque exige que todas las características sean discretas, condición que no se cumple con columnas continuas como `amount` o las agregaciones de ventana temporal.
# MAGIC 8. **Escalado** (`StandardScaler`): escala el vector de características. Los modelos basados en árboles son invariantes al escalado, por lo que este paso no les aporta ninguna mejora. Sin embargo, lo mantenemos en el *pipeline* base por simplicidad: el coste adicional es asumible y garantiza que todos los modelos reciben exactamente el mismo preprocesado.

# COMMAND ----------

preprocessing_stages = [
    imputer,  # 1. Median imputation for profile and null aggregation columns
    boolean_transformer,  # 2. Cast boolean flags to Double
    feature_engineer,  # 3. Create new features from current transaction fields
    string_indexer,  # 4. Learn category vocabulary
    ohe,  # 5. Convert indices to sparse vectors
    assembler,  # 6. Assemble all columns into feature vector
    var_selector,  # 7. Remove quasi-constant features (variance threshold)
    standard_scaler  # 8. Scale features
]

preprocessing_pipeline = Pipeline(stages = preprocessing_stages)

for i, stage in enumerate(preprocessing_stages, 1):
    print(f"{i}. {type(stage).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Partición: entrenamiento, validación y prueba
# MAGIC
# MAGIC La división del conjunto de datos es uno de los pasos más críticos del flujo de aprendizaje automático. En un problema de detección de fraude con datos temporales, **no se puede usar `randomSplit`**: dividir aleatoriamente las filas mezclaría transacciones del futuro en el entrenamiento con transacciones del pasado en la prueba, lo que introduciría *data leakage* y produciría métricas de evaluación irrealmente optimistas.
# MAGIC
# MAGIC La división correcta es **temporal**: el modelo debe entrenarse únicamente con transacciones anteriores a la fecha de evaluación, exactamente igual que ocurriría en producción. Esto garantiza que el modelo nunca «ve» el futuro durante el entrenamiento.
# MAGIC
# MAGIC ### Estrategia de partición para este seminario
# MAGIC
# MAGIC | Conjunto | Período | Propósito |
# MAGIC |---|---|---|
# MAGIC | Entrenamiento | Hasta finales de 2022 | Ajustar los parámetros del modelo (`Pipeline.fit`) |
# MAGIC | Validación | 2023 | Seleccionar hiperparámetros |
# MAGIC | Prueba | 2024 | Evaluación final, solo se toca **una vez** al final |
# MAGIC
# MAGIC Esta estrategia tiene además una consecuencia importante para la **validación cruzada**: tanto el `CrossValidator` como el `TrainValidationSplit` de `MLlib` realizan divisiones aleatorias de los datos por defecto, lo que los hace **incompatibles** con series temporales, dado que el pasado no puede conocer el futuro. Dado que `MLlib` no implementa de forma nativa una validación cruzada temporal (*time series split*), en este seminario evitaremos el uso de estas herramientas para realizar la partición. En su lugar, realizaremos una **división cronológica manual** utilizando la columna `timestamp`, garantizando así que el conjunto de validación y prueba sean siempre posteriores al de entrenamiento, evitando cualquier ***data leakage* temporal**.
# MAGIC
# MAGIC > **Documentación oficial**: [`ML Tuning`](https://spark.apache.org/docs/latest/ml-tuning.html)
# MAGIC

# COMMAND ----------

# Filter out rows with null label.
# Should already be clean in the Delta table, but we add this guard just in case.
df_labeled = df_raw.filter(F.col(label_column).isNotNull())

# Cast the label to double because it is required by the library
df_labeled = df_labeled.withColumn(label_column, F.col(label_column).cast(DoubleType()))

total_rows = df_labeled.count()
print(f"Labeled rows available: {total_rows:,}")

# Temporal split: train on historical data, validate and test on more recent periods.
# Using filter on the timestamp column ensures no future data leaks into training.
train_df = df_labeled.filter(F.col("timestamp") <= "2022-12-31")
validation_df = df_labeled.filter((F.col("timestamp") >= "2023-01-01") & (F.col("timestamp") <= "2023-12-31"))
test_df = df_labeled.filter(F.col("timestamp") >= "2024-01-01")

train_count = train_df.count()
validation_count = validation_df.count()
test_count = test_df.count()

print(f"Training dataset : {train_count:,} rows; ({100 * train_count / total_rows:.1f}%) up to 2022")
print(f"Validation dataset: {validation_count:,} rows; ({100 * validation_count / total_rows:.1f}%) in 2023")
print(f"Testing dataset: {test_count:,} rows; ({100 * test_count / total_rows:.1f}%) in 2024")

# COMMAND ----------

# Verify that the class balance is consistent across partitions.
# With a temporal split, the fraud rate may vary between periods,
# which is expected and reflects real-world distribution shifts over time.
for name, partition in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
    total = partition.count()
    fraud = partition.filter(F.col(label_column) == 1.0).count()
    print(f"{name}: {fraud:,} frauds out of {total:,} ({100 * fraud / total:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 7. Gestión del desequilibrio de clases
# MAGIC
# MAGIC Con una tasa de fraude de entre el 2% y el 4% según el período, si el modelo aprende a predecir siempre «legítima» obtendrá más del 96% de exactitud (*accuracy*) sin detectar un solo fraude. La exactitud no es la métrica adecuada; tampoco lo es entrenar sin compensar el desequilibrio.
# MAGIC
# MAGIC Cabe destacar que la tasa de fraude no es constante entre particiones (2.44% en entrenamiento, 3.44% en validación y 3.91% en prueba). Este incremento progresivo es coherente con la realidad: los patrones de fraude evolucionan con el tiempo y los defraudadores adaptan sus técnicas. Es una señal de *distribution shift* real que el modelo deberá ser capaz de generalizar, y que hace que las métricas sobre el conjunto de prueba sean más exigentes que las del entrenamiento.
# MAGIC
# MAGIC ### 7.1. Pesos por clase (`weightCol`)
# MAGIC
# MAGIC La mayoría de los clasificadores de `MLlib` aceptan un parámetro `weightCol` que indica qué columna del `DataFrame` contiene el peso de cada muestra. Podemos calcular estos pesos de forma que cada clase contribuya igual al gradiente total.
# MAGIC
# MAGIC Esta fórmula es el equivalente al parámetro `class_weight = "balanced"` de `scikit-learn`.
# MAGIC
# MAGIC ### 7.2. *Oversampling* del conjunto de entrenamiento
# MAGIC
# MAGIC Duplicar o sobremuestrear la clase minoritaria es equivalente a asignar pesos, pero puede saturar la memoria del *driver* al hacer `.collect()`. Con `Spark`, la alternativa es hacer una `union` del `DataFrame` de fraudes con sí mismo varias veces, lo que mantiene la distribución en el clúster.
# MAGIC
# MAGIC ### 7.3. Ajuste del umbral de decisión (`thresholds`)
# MAGIC
# MAGIC Los clasificadores de `MLlib` exponen el parámetro `thresholds` para ajustar el umbral de decisión por clase. Bajar el umbral para la clase positiva (fraude) aumenta la sensibilidad (*recall*) a costa de más falsos positivos (*precision*). El umbral óptimo se determina analizando la curva *precision-recall*, y dado que la tasa de fraude en producción es superior a la del entrenamiento, es especialmente importante calibrar este umbral sobre el conjunto de validación antes de evaluar sobre el de prueba.

# COMMAND ----------

# Compute class weights from the training set only.
# Using training counts ensures no information from validation or test leaks in.
n_train = train_df.count()
n_fraud = train_df.filter(F.col(label_column) == 1.0).count()
n_legit = n_train - n_fraud

weight_fraud = n_train / (2.0 * n_fraud)
weight_legit = n_train / (2.0 * n_legit)

print(f"Total rows: {n_train:,}")
print(f"Fraud: {n_fraud:,} ({100 * n_fraud / n_train:.2f}%)")
print(f"Legitimate: {n_legit:,} ({100 * n_legit / n_train:.2f}%)")
print(f"Fraud weight: {weight_fraud:.4f}")
print(f"Legit weight: {weight_legit:.4f}")

# Add the class weight column to the training set.
# Validation and test sets do not need weights: they are only used for evaluation.
class_weight_column = "class_weight"

train_weighted = train_df.withColumn(
    class_weight_column,
    F.when(F.col(label_column) == 1.0, weight_fraud).otherwise(weight_legit)
)

train_weighted.groupBy(label_column).agg(
    F.count("*").alias("n"),
    F.avg(class_weight_column).alias("avg_weight")
).orderBy(label_column).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 8. Entrenamiento de modelos de clasificación
# MAGIC
# MAGIC En esta sección entrenamos tres modelos distintos integrando el preprocesado y el clasificador en un único `Pipeline`. La lógica es siempre la misma:
# MAGIC
# MAGIC 1. Definir el clasificador con sus hiperparámetros.
# MAGIC 2. Construir un `Pipeline` que encadena las etapas de preprocesado con el clasificador.
# MAGIC 3. Llamar a `pipeline.fit(train_df)` para ajustar todo en orden: cada `Estimator` del preprocesado aprende sus parámetros (vocabularios, medianas, escalas) y el clasificador aprende los suyos (coeficientes, estructura de árboles).
# MAGIC 4. Llamar a `model.transform(test_df)` para obtener las predicciones. El `DataFrame` resultante incluye las columnas `rawPrediction`, `probability` y `prediction`.
# MAGIC
# MAGIC > **Documentación oficial**: [Classification and regression](https://spark.apache.org/docs/latest/ml-classification-regression.html)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 8.1. Regresión logística (`LogisticRegression`)
# MAGIC
# MAGIC La regresión logística es el modelo lineal de referencia para clasificación binaria.
# MAGIC
# MAGIC Hiperparámetros clave:
# MAGIC
# MAGIC * `maxIter`: número máximo de iteraciones del optimizador.
# MAGIC * `regParam` (λ): fuerza de la regularización (mayor → modelo más simple → más sesgo, menos varianza).
# MAGIC * `elasticNetParam` (α): mezcla entre L1 (selección de características) y L2 (contracción de coeficientes). `0` = solo L2 (Ridge), `1` = solo L1 (Lasso).
# MAGIC * `weightCol`: columna con el peso de cada muestra para gestionar el desequilibrio.
# MAGIC * `featuresCol` y `labelCol`: nombres estándar de las columnas de entrada y salida.
# MAGIC
# MAGIC > **Documentación oficial**: [`LogisticRegression`](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression)
# MAGIC

# COMMAND ----------

lr_features_col = scaler_output_column
lr_label_col = label_column
lr_weight_col = class_weight_column
lr_max_iter = 100
lr_reg_param = 0.01  # L2 regularization strength
lr_elastic_net_param = 0.0  # 0.0 = pure L2 (Ridge), 1.0 = pure L1 (Lasso)
lr_family = "binomial"
lr_standardization = False  # Already scaled in the pipeline

lr = LogisticRegression(
    featuresCol = lr_features_col,
    labelCol = lr_label_col,
    weightCol = lr_weight_col,
    maxIter = lr_max_iter,
    regParam = lr_reg_param,
    elasticNetParam = lr_elastic_net_param,
    family = lr_family,
    standardization = lr_standardization
)

lr_stages = preprocessing_stages + [lr]
lr_pipeline = Pipeline(stages = lr_stages)

print("Training logistic regression.")
lr_model = lr_pipeline.fit(train_weighted)
print("Training complete.")

# COMMAND ----------

lr_predictions = lr_model.transform(test_df)

lr_predictions.select(
    label_column,
    raw_prediction_column,
    probability_column,
    prediction_column
).limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 8.2. *Random forests* (`RandomForestClassifier`)
# MAGIC
# MAGIC *Random forests* es un ensemble de árboles de decisión entrenados de forma independiente sobre subconjuntos aleatorios del dato (*bootstrap sampling*) y con subconjuntos aleatorios de características en cada nodo. La predicción final es la votación mayoritaria de todos los árboles.
# MAGIC
# MAGIC Ventajas para detección de fraude:
# MAGIC
# MAGIC * Muy robusto ante características irrelevantes y ruido.
# MAGIC * Captura interacciones no lineales entre características.
# MAGIC * **No necesita escalado de características**: sus decisiones se basan en umbrales sobre características individuales, no en distancias. El escalado que incluimos en el *pipeline* base no le perjudica, pero tampoco le aporta nada.
# MAGIC * La importancia de características (`featureImportances`) es directamente interpretable: indica qué señales han sido más determinantes para las decisiones del modelo.
# MAGIC
# MAGIC Hiperparámetros clave:
# MAGIC * `numTrees`: número de árboles en el ensemble. Más árboles → mayor estabilidad pero más tiempo de entrenamiento.
# MAGIC * `maxDepth`: profundidad máxima de cada árbol. Árboles más profundos → mayor capacidad de capturar patrones complejos pero mayor riesgo de sobreajuste.
# MAGIC * `maxBins`: número de divisiones consideradas al evaluar cada característica en un nodo. Afecta la resolución con la que el árbol puede ubicar los umbrales óptimos.
# MAGIC * `featureSubsetStrategy`: fracción de características a considerar en cada nodo. `"sqrt"` es el valor canónico para clasificación.
# MAGIC
# MAGIC > **Documentación oficial**: [`RandomForestClassifier`](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier)
# MAGIC

# COMMAND ----------

rf_features_col = scaler_output_column
rf_label_col = label_column
rf_weight_col = class_weight_column
rf_num_trees = 5  # Kept low for the seminar: increase for production
rf_max_depth = 4  # Kept low for the seminar: increase for production
rf_max_bins = 32
rf_feature_subset_strategy = "sqrt"
rf_seed = seed

rf = RandomForestClassifier(
    featuresCol = rf_features_col,
    labelCol = rf_label_col,
    weightCol = rf_weight_col,
    numTrees = rf_num_trees,
    maxDepth = rf_max_depth,
    maxBins = rf_max_bins,
    featureSubsetStrategy = rf_feature_subset_strategy,
    seed = rf_seed
)

rf_stages = preprocessing_stages + [rf]
rf_pipeline = Pipeline(stages = rf_stages)

print("Training random forests.")
rf_model = rf_pipeline.fit(train_weighted)
print("Training complete.")

# Extract the random forests model from the pipeline to access feature importances
rf_stage = rf_model.stages[-1]
importances = rf_stage.featureImportances
print(f"\nFeature importances (first 10): {list(importances.toArray()[:10])}")

# COMMAND ----------

rf_predictions = rf_model.transform(test_df)

rf_predictions.select(
    label_column,
    raw_prediction_column,
    probability_column,
    prediction_column
).limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 8.3. *Gradient boosting* (`GBTClassifier`)
# MAGIC
# MAGIC *Gradient boosting* entrena árboles **secuencialmente**: cada árbol aprende a corregir los errores del árbol anterior. El resultado es un modelo con menor sesgo que *random forests* pero más propenso al sobreajuste y más sensible a los hiperparámetros.
# MAGIC
# MAGIC En la práctica, *gradient boosting* suele superar a *random forests* en conjuntos de datos tabulares con muchas características, lo que lo convierte en el algoritmo de referencia para sistemas de detección de fraude en producción.
# MAGIC
# MAGIC Hiperparámetros clave:
# MAGIC
# MAGIC * `maxIter`: número de árboles (= número de iteraciones de *boosting*). Más es mejor hasta cierto punto; después hay sobreajuste.
# MAGIC * `stepSize` (tasa de aprendizaje): factor por el que se multiplica la contribución de cada nuevo árbol. Valores pequeños (entre 0.05 y 0.1) con muchos árboles suelen ser más robustos.
# MAGIC * `maxDepth`: árboles más profundos capturan interacciones de orden mayor.
# MAGIC * `subsamplingRate`: fracción del dato a usar en cada iteración. Introduce aleatoriedad y reduce sobreajuste (similar al *stochastic gradient boosting* de *extreme gradient boosting*).
# MAGIC
# MAGIC > **Limitación actual**: la implementación de `GBTClassifier` en `MLlib` **no soporta `weightCol`**. Para gestionar el desequilibrio, la alternativa es aplicar *oversampling* de la clase minoritaria directamente en `Spark`, o ajustar manualmente el `thresholds` en la predicción posterior.
# MAGIC
# MAGIC > **Documentación oficial**: [`GBTClassifier`](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier)
# MAGIC

# COMMAND ----------

gbt_features_col = scaler_output_column
gbt_label_col = label_column
gbt_max_iter = 5  # Kept low for the seminar: increase for production
gbt_step_size = 0.1
gbt_max_depth = 4  # Kept low for the seminar: increase for production
gbt_subsampling_rate = 0.8
gbt_seed = seed

# Class imbalance is handled by adjusting the decision threshold during evaluation instead.
gbt = GBTClassifier(
    featuresCol = gbt_features_col,
    labelCol = gbt_label_col,
    maxIter = gbt_max_iter,
    stepSize = gbt_step_size,
    maxDepth = gbt_max_depth,
    subsamplingRate = gbt_subsampling_rate,
    seed = gbt_seed
)

gbt_stages = preprocessing_stages + [gbt]
gbt_pipeline = Pipeline(stages = gbt_stages)

print("Training gradient boosting.")
gbt_model = gbt_pipeline.fit(train_df)
print("Training complete.")

# COMMAND ----------

gbt_predictions = gbt_model.transform(test_df)

gbt_predictions.select(
    label_column,
    raw_prediction_column,
    probability_column,
    prediction_column
).limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 9. Evaluación de modelos
# MAGIC
# MAGIC Dado el fuerte desequilibrio de clases que ya hemos visto (2.44% en entrenamiento, 3.44% en validación y 3.91% en prueba), la **exactitud** (*accuracy*) es una métrica engañosa: un modelo que nunca predice fraude obtiene más del 96% de exactitud sin detectar un solo caso positivo.
# MAGIC
# MAGIC | Métrica | Descripción | Relevancia para fraude |
# MAGIC |---|---|---|
# MAGIC | ***AUC-ROC*** | Área bajo la curva `ROC` (`TPR` vs. `FPR`) | Mide la capacidad discriminante global del modelo, independiente del umbral. |
# MAGIC | ***AUC-PR*** | Área bajo la curva *precision-recall* | Más informativa que *AUC-ROC* cuando la clase positiva es minoritaria. Con un más o menos 3% de fraudes, un modelo aleatorio obtendría un *AUC-PR* de ~0.03; cualquier valor significativamente superior indica capacidad discriminante real. |
# MAGIC | ***Precision*** | `TP / (TP + FP)` | De todas las transacciones marcadas como fraude, ¿cuántas son realmente fraude? Alta *precision* = pocas falsas alarmas. |
# MAGIC | ***Recall*** | `TP / (TP + FN)` | De todos los fraudes reales, ¿cuántos detectamos? Alto *recall* = pocas pérdidas. |
# MAGIC | ***F1-score*** | `2 × (P × R) / (P + R)` | Media armónica de *precision* y *recall*. Útil como métrica de referencia única. |
# MAGIC
# MAGIC El balance entre *precision* y *recall* depende de los costes de negocio: un falso negativo (fraude no detectado) tiene un coste mucho mayor que un falso positivo (transacción legítima bloqueada). En la práctica, los sistemas de fraude bancario suelen optimizar el ***recall*** con una restricción de precisión mínima (por ejemplo, «detectar al menos el 85% de los fraudes con un máximo de 10% de falsos positivos»).
# MAGIC
# MAGIC `MLlib` proporciona dos evaluadores:
# MAGIC
# MAGIC * `BinaryClassificationEvaluator`: *AUC-ROC* y *AUC-PR*. Opera sobre `rawPrediction`, no sobre `prediction`, lo que lo hace independiente del umbral de decisión.
# MAGIC * `MulticlassClassificationEvaluator`: exactitud, *precision*, *recall* y *F1-score*. Opera sobre `prediction`, por lo que sus resultados dependen del umbral de decisión (0.5 por defecto).

# COMMAND ----------

# Binary evaluator: AUC-ROC and AUC-PR.
# Operates on raw predictions, so results are independent of the decision threshold.
evaluator_auc = BinaryClassificationEvaluator(
    labelCol = label_column,
    rawPredictionCol = raw_prediction_column,
    metricName = "areaUnderROC"
)

evaluator_pr = BinaryClassificationEvaluator(
    labelCol = label_column,
    rawPredictionCol = raw_prediction_column,
    metricName = "areaUnderPR"
)

# Multiclass evaluator: precision, recall, and F1-score.
# Operates on prediction, so results depend on the decision threshold (0.5 by default).
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "f1"
)

evaluator_precision = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "weightedPrecision"
)

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "weightedRecall"
)

# COMMAND ----------

def evaluate_model(predictions, model_name):
    """Computes and displays all evaluation metrics for a set of predictions."""
    auc_roc = evaluator_auc.evaluate(predictions)
    auc_pr = evaluator_pr.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    return {
        "model": model_name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# COMMAND ----------

# Evaluate all three models on the test set and collect results for comparison.
results = []
results.append(evaluate_model(lr_predictions, "Logistic regression"))
results.append(evaluate_model(rf_predictions, "Random forests"))
results.append(evaluate_model(gbt_predictions, "Gradient boosting"))

# Comparative results table across all three models.
pd.DataFrame(results).set_index("model").round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 10. Ajuste de hiperparámetros
# MAGIC
# MAGIC El ajuste manual de hiperparámetros es tedioso y no garantiza encontrar la combinación óptima. `MLlib` proporciona dos mecanismos automáticos: `CrossValidator` y `TrainValidationSplit`.
# MAGIC
# MAGIC Sin embargo, ambos tienen una limitación crítica en nuestro caso: **dividen el dato de forma aleatoria**, lo que es incompatible con series temporales.
# MAGIC
# MAGIC Por tanto, esta sección se presenta como **referencia sin código**: muestra cómo se usarían `CrossValidator` y `TrainValidationSplit` en un entorno donde `MLlib`.
# MAGIC
# MAGIC > **Documentación oficial**: [`ML Tuning`](https://spark.apache.org/docs/latest/ml-tuning.html)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 10.1. Validación cruzada *k*-*folds* (`CrossValidator`)
# MAGIC
# MAGIC `CrossValidator` divide el conjunto de entrenamiento en *k folds* de forma estratificada. En cada iteración, *k* - 1 *folds* se usan para entrenar y el *fold* restante para evaluar. El resultado es la media de las *k* métricas de evaluación, lo que produce una estimación más robusta del rendimiento generalizable que una única división.
# MAGIC
# MAGIC El coste computacional es prohibitivo con 60 millones de instancias. En ese caso, la alternativa práctica es `TrainValidationSplit`.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 10.2. División en entrenamiento y validación (`TrainValidationSplit`)
# MAGIC
# MAGIC `TrainValidationSplit` evalúa cada combinación de hiperparámetros una única vez sobre un subconjunto de validación generado dividiendo aleatoriamente el dato según el parámetro `trainRatio`. Es más rápido que `CrossValidator` pero produce una estimación menos estable del rendimiento.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 11. Catálogo de técnicas disponibles en `MLlib`
# MAGIC
# MAGIC Esta sección actúa como guía de referencia: recoge las principales técnicas disponibles en cada módulo de `MLlib`, con una breve descripción y el enlace directo a la documentación oficial. El objetivo es que el alumno sepa **dónde buscar** cuando necesite una técnica concreta.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.1. Transformadores de características
# MAGIC
# MAGIC [Documentación completa: `ml-features`](https://spark.apache.org/docs/latest/ml-features.html)
# MAGIC
# MAGIC | Transformador | Descripción |
# MAGIC |---|---|
# MAGIC | `Tokenizer` y `RegexTokenizer` | Divide cadenas de texto en *tokens* |
# MAGIC | `HashingTF` y `IDF` | `TF-IDF` sobre texto *tokenizado* |
# MAGIC | `Word2Vec` | *Embeddings* de palabras |
# MAGIC | `PCA` | Reducción de dimensionalidad |
# MAGIC | `DCT` | Transformada discreta del coseno |
# MAGIC | `PolynomialExpansion` | Expansión polinómica de características |
# MAGIC | `Binarizer` | Umbralización de una columna continua |
# MAGIC | `ElementwiseProduct` | Producto elemento a elemento de dos vectores |
# MAGIC | `Interaction` | Producto cartesiano de dos vectores de características |
# MAGIC | `RFormula` | Especificación de modelos al estilo `R` |
# MAGIC | `IndexToString` | Convierte un índice de vuelta a la categoría original |
# MAGIC | `TargetEncoder` | Codificación de categóricas por media de la etiqueta |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.2. Algoritmos de clasificación
# MAGIC
# MAGIC [Documentación completa: `ml-classification-regression`](https://spark.apache.org/docs/latest/ml-classification-regression.html)
# MAGIC
# MAGIC | Algoritmo | Descripción |
# MAGIC |---|---|
# MAGIC | `LogisticRegression` | Clasificación lineal binaria o multinomial |
# MAGIC | `DecisionTreeClassifier` | Árbol de decisión individual |
# MAGIC | `RandomForestClassifier` | *Ensemble* de árboles con *bootstrap aggregating* |
# MAGIC | `GBTClassifier` | *Ensemble* secuencial con *gradient boosting* |
# MAGIC | `LinearSVC` | Máquinas vectores soporte lineal |
# MAGIC | `NaiveBayes` | Clasificador probabilístico Bayesiano |
# MAGIC | `MultilayerPerceptronClassifier` | Red neuronal *feed-forward* |
# MAGIC | `FMClassifier` | Máquinas de factorización |
# MAGIC | `OneVsRest` | Estrategia uno-contra-todos para multiclase |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.3. Algoritmos de regresión
# MAGIC
# MAGIC [Documentación completa: `ml-classification-regression`](https://spark.apache.org/docs/latest/ml-classification-regression.html)
# MAGIC
# MAGIC | Algoritmo | Descripción |
# MAGIC |---|---|
# MAGIC | `LinearRegression` | Regresión lineal con regularización L1, L2 o *elastic net* |
# MAGIC | `DecisionTreeRegressor` | Árbol de decisión para regresión |
# MAGIC | `RandomForestRegressor` | *Ensemble* de árboles con *bootstrap aggregating* para regresión |
# MAGIC | `GBTRegressor` | *Ensemble* secuencial con *gradient booting* para regresión |
# MAGIC | `GeneralizedLinearRegression` | Modelos lineales generalizados con distintas familias de distribución (Poisson, Gamma, etc.) |
# MAGIC | `AFTSurvivalRegression` | Regresión de supervivencia con tiempos de fallo |
# MAGIC | `IsotonicRegression` | Regresión isotónica (monótona no decreciente) |
# MAGIC | `FMRegressor` | Máquinas de factorización para regresión |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.4. Algoritmos de *clustering* (no supervisado)
# MAGIC
# MAGIC [Documentación completa: `ml-clustering`](https://spark.apache.org/docs/latest/ml-clustering.html)
# MAGIC
# MAGIC | Algoritmo | Descripción |
# MAGIC |---|---|
# MAGIC | `KMeans` | Agrupamiento por centroides |
# MAGIC | `BisectingKMeans` | *k*-medias divisivo |
# MAGIC | `GaussianMixture` | Mezcla de Gaussianas |
# MAGIC | `LDA` | Asignación latente de Dirichlet |
# MAGIC | `PowerIterationClustering` | *Clustering* espectral escalable |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.5. Detección de anomalías e *isolation*
# MAGIC
# MAGIC [Documentación completa: `ml-advanced`](https://spark.apache.org/docs/latest/ml-advanced.html)
# MAGIC
# MAGIC `MLlib` no incluye *isolation forests* de forma nativa. Sin embargo, hay técnicas aplicables:
# MAGIC
# MAGIC * **`GaussianMixture`**: modela la distribución «normal» de los datos y puntúa las desviaciones respecto a ella.
# MAGIC * ***Autoencoders***: no disponibles en `MLlib`; requieren `PyTorch` o `TensorFlow`.
# MAGIC * **Percentil de probabilidad inverso**: con cualquier clasificador, el percentil inferior de `probability[1]` sobre el conjunto de datos no etiquetado actúa como puntuación de anomalía.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.6. Estadísticas
# MAGIC
# MAGIC [Documentación completa: `ml-statistics`](https://spark.apache.org/docs/latest/ml-statistics.html)
# MAGIC
# MAGIC | Función | Descripción |
# MAGIC |---|---|
# MAGIC | `Summarizer` | Media, varianza, mínimo, máximo y norma de un vector |
# MAGIC | `Correlation` | Matriz de correlación de Pearson o Spearman |
# MAGIC | `ChiSquareTest` | Test `χ²` entre cada característica y la etiqueta |
# MAGIC | `KolmogorovSmirnovTest` | Test `KS` de bondad de ajuste a una distribución |
# MAGIC | `ANOVATest` | F-test ANOVA entre una característica continua y etiquetas |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11.7. Fuentes de datos compatibles con `MLlib`
# MAGIC
# MAGIC [Documentación completa: `ml-datasource`](https://spark.apache.org/docs/latest/ml-datasource.html)
# MAGIC
# MAGIC Además de leer desde tablas `Delta` como hacemos en este seminario, `MLlib` puede cargar directamente varios formatos estándar del ecosistema de aprendizaje automático:
# MAGIC
# MAGIC * **`LIBSVM`**: formato esparso usado en `libsvm` y `XGBoost`. `spark.read.format("libsvm").load(path)`.
# MAGIC * **`Image`**: directorio de imágenes para aprendizaje profundo con `spark.read.format("image").load(path)`.
# MAGIC * **`Audio`**: ficheros de audio (con la librería `spark-audio`).
# MAGIC
# MAGIC Para nuestro caso de uso, la lectura desde `Delta` es siempre la opción óptima por su integración con `Unity Catalog`, soporte de *time travel* y rendimiento en lectura con *predicate pushdown*.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 12. Conclusiones
# MAGIC
# MAGIC ### ¿Qué hemos visto en este seminario?
# MAGIC
# MAGIC Este seminario ha cubierto de extremo a extremo el uso de `MLlib` para detección de fraude:
# MAGIC
# MAGIC 1. **Carga de datos desde `Delta`**: `spark.table()` devuelve un plan distribuido, no una tabla en memoria. Todas las transformaciones se ejecutan en los *workers*.
# MAGIC
# MAGIC 2. **Conceptos fundamentales**: `Transformer` (sin estado, `.transform()`), `Estimator` (aprende del dato, `.fit()` devuelve un `Model`), `Pipeline` (cadena reproducible que previene *data leakage*).
# MAGIC
# MAGIC 3. **Preprocesado completo**:
# MAGIC    * Imputación con `Imputer` (mediana para perfil y agregaciones estadísticas nulas).
# MAGIC    * Codificación de categóricas con `StringIndexer` + `OneHotEncoder` (`handleInvalid = "keep"` para producción).
# MAGIC    * Ingeniería de características con `SQLTransformer` (nuevas señales a partir de los campos de la transacción actual).
# MAGIC    * Selección con `VarianceThresholdSelector` y `UnivariateFeatureSelector` antes del escalado.
# MAGIC    * Escalado con `StandardScaler` sobre el vector ya reducido.
# MAGIC
# MAGIC 4. **Partición temporal**: división por fecha en lugar de `randomSplit`, imprescindible con series temporales para evitar *data leakage*.
# MAGIC
# MAGIC 5. **Desequilibrio de clases**: `weightCol` en `LogisticRegression` y `RandomForestClassifier`; para `GBTClassifier`, ajuste del umbral de decisión.
# MAGIC
# MAGIC 6. **Tres modelos**: Regresión logística, *random forests* y *gradient boosting*, cada uno con su `Pipeline` completo de preprocesado + clasificador.
# MAGIC
# MAGIC 7. **Evaluación correcta**: *AUC-ROC*, *AUC-PR*, *precision*, *recall* y *F1-score*. La exactitud es una métrica engañosa con desequilibrio de clases.
# MAGIC
# MAGIC 8. **Ajuste de hiperparámetros**: `CrossValidator` y `TrainValidationSplit`, con la advertencia de que ambos son incompatibles con series temporales por su división aleatoria.
# MAGIC
# MAGIC ### ¿Cómo encaja este seminario con el flujo principal de la asignatura?
# MAGIC
# MAGIC | Aspecto | Flujo principal | Este seminario |
# MAGIC |---|---|---|
# MAGIC | Carga de datos | `FeatureEngineeringClient` + `PiT` *join* | `spark.table()` directamente sobre `Delta` |
# MAGIC | Preprocesado | `MLlib Pipeline` | `MLlib Pipeline` (idéntico) |
# MAGIC | Seguimiento | `MLflow` (métricas, parámetros, artefactos) | No (omitido) |
# MAGIC | Trazabilidad | `delta_version` + `MLflow` run | No (omitido) |
# MAGIC | Servicio | `MLflow Model Serving` + `Online Feature Store` | No (omitido) |
# MAGIC | Reentrenamiento | `Lakeflow` job automatizado | No (omitido) |
# MAGIC
# MAGIC El `Pipeline` de preprocesado que hemos construido en este seminario **es exactamente el mismo** que se usaría en el flujo principal: la única diferencia es que allí se registra como artefacto de `MLflow` y se despliega detrás de la `API` de *serving*. Todo lo aprendido aquí es directamente transferible.
# MAGIC
# MAGIC ### Recursos adicionales
# MAGIC
# MAGIC - [Guía completa de `MLlib`](https://spark.apache.org/docs/latest/ml-guide.html)
# MAGIC - [`API Reference PySpark ML`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html)
# MAGIC - [`MLlib Statistics`](https://spark.apache.org/docs/latest/ml-statistics.html)
# MAGIC - [`MLlib Data Sources`](https://spark.apache.org/docs/latest/ml-datasource.html)
# MAGIC - [`MLlib Pipelines`](https://spark.apache.org/docs/latest/ml-pipeline.html)
# MAGIC - [`MLlib Features`](https://spark.apache.org/docs/latest/ml-features.html)
# MAGIC - [`MLlib Classification & Regression`](https://spark.apache.org/docs/latest/ml-classification-regression.html)
# MAGIC - [`MLlib Tuning`](https://spark.apache.org/docs/latest/ml-tuning.html)
# MAGIC - [`MLlib Advanced`](https://spark.apache.org/docs/latest/ml-advanced.html)
# MAGIC