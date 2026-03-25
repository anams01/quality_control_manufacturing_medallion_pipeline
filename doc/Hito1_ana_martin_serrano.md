---
layout: none
---

<style>
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 12pt;
    line-height: 1.5;
    margin: 3cm 2.5cm 3cm 2.5cm;
}

h1, h2, h3 {
    font-family: "Times New Roman", Times, serif;
    font-weight: bold;
    color: #1a1a1a;
}

h1 { font-size: 20pt; margin-bottom: 0.5em; }
h2 { font-size: 16pt; margin-top: 1em; margin-bottom: 0.4em; }
h3 { font-size: 14pt; margin-top: 0.8em; margin-bottom: 0.3em; }

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1em;
    margin-bottom: 1em;
}

table, th, td {
    border: 1px solid #999;
}

th, td {
    padding: 6px 10px;
    text-align: center;
    font-size: 11pt;
}

th {
    background-color: #f2f2f2;
}

div.portada {
    text-align: center;
    margin-top: 150px;
}

div.portada h1 {
    font-size: 22pt;
}

div.portada p {
    font-size: 12pt;
    margin: 0.4em 0;
}

.page-break {
    page-break-after: always;
}
</style>

<div class="portada">

# Hito 1: Control de calidad en manufactura

**Asignatura:** Desarrollo y despliegue de soluciones Big Data  
**Máster:** Big Data y Análisis de Datos  

<br>

**Estudiante:** Ana Martín Serrano  

<br>

**Fecha de entrega:** Febrero 2026

</div>

<div class="page-break"></div>

---

# Alcance y viabilidad

## Definición del problema de negocio

La planta de producción de componentes electrónicos fabrica un volumen masivo de **1.000.000 de unidades mensuales**. A pesar de los controles automatizados actuales, la línea se enfrenta a una tasa de defectos del **2% (20.000 piezas defectuosas)**, un margen de error que compromete la reputación de la marca ante sus clientes.

Esta imprecisión en el filtrado de calidad genera pérdidas millonarias por dos vías opuestas.

Por un lado, la falta de sensibilidad de los sensores actuales permite que lleguen al mercado **8.000 componentes defectuosos al mes** (dejando escapar el 40% de los defectos reales). Dado que el coste de gestión de la garantía, devolución y penalización por pieza defectuosa es de **200 euros**, la compañía asume pérdidas directas de **1.600.000 euros mensuales**.

Por otro lado, la calibración excesivamente cautelosa de las máquinas desecha componentes que son perfectamente funcionales. Actualmente, se envían a la trituradora el **5% de la producción válida (49.000 unidades)**. Con un coste de fabricación perdido de **10 euros por unidad**, se desperdician **490.000 euros** en material y energía.

El impacto financiero total de esta ineficiencia asciende a **2.090.000 euros al mes**.

---

## Planteamiento y selección de la solución técnica

Para determinar la arquitectura óptima, el equipo técnico sometió a evaluación tres escenarios posibles, atendiendo tanto a la complejidad del problema como a las restricciones operativas del entorno industrial.

En primera instancia, se descartó la optimización del sistema heurístico actual, basado en reglas fijas y calibración manual de sensores. La experiencia acumulada en planta ha demostrado que este enfoque resulta operativamente insostenible y es incapaz de mejorar la calidad global del proceso sin provocar efectos colaterales indeseados. En concreto, cualquier ajuste orientado a incrementar la detección de defectos reales incrementa de forma inmediata la tasa de falsos positivos, elevando el descarte de piezas válidas y el desperdicio de material. Esta rigidez estructural impide alcanzar un equilibrio económico aceptable mediante reglas manuales.

Posteriormente, se analizó la viabilidad de una solución de aprendizaje automático tradicional desplegada sobre un servidor monolítico mediante escalado vertical. Si bien esta alternativa podría simplificar el desarrollo inicial, presenta riesgos estructurales significativos en el contexto del problema planteado. Por un lado, la necesidad de entrenar modelos complejos utilizando ventanas históricas plurianuales, indispensables para capturar patrones de degradación progresiva de maquinaria y efectos estacionales, exigiría cargar en memoria volúmenes de datos incompatibles con un servidor convencional, limitando la iteración y el reentrenamiento. Por otro lado, una arquitectura centralizada carece de la elasticidad necesaria para absorber incrementos de carga derivados de picos de producción o ampliaciones futuras de la planta, comprometiendo la estabilidad operativa del sistema.

Por consiguiente, la elección final recae sobre una arquitectura Big Data distribuida. Esta decisión es la única que garantiza el cumplimiento de las variables críticas del proyecto: capacidad de cómputo paralelo para procesar el volumen histórico requerido, escalabilidad horizontal para mantener el rendimiento ante variaciones en la producción, y flexibilidad suficiente para adaptar los modelos a la variabilidad no lineal de los patrones de defecto propios de un entorno industrial complejo

---

## Evaluación de la viabilidad y valor

El análisis de factibilidad ratifica la solidez del proyecto en sus tres dimensiones críticas: viabilidad técnica, viabilidad económica y viabilidad ética y legal.

En el plano de la viabilidad técnica, la organización parte de una posición favorable al disponer de un repositorio histórico saneado que abarca los últimos cinco años de operativa industrial. Este activo de datos, que suma aproximadamente 60 millones de registros etiquetados a nivel de unidad fabricada, garantiza el volumen necesario para entrenar modelos supervisados sin riesgo significativo de sobreajuste. La información disponible incluye señales de sensores industriales, datos de línea de producción y resultados finales de inspección, lo que permite capturar tanto patrones instantáneos como efectos de degradación progresiva de la maquinaria.

Desde la perspectiva de la viabilidad económica, la proyección es igualmente sólida. Partimos de unas pérdidas actuales estimadas en 2.090.000 euros mensuales, desglosadas en 1.600.000 euros derivados de la salida al mercado de componentes defectuosos y 490.000 euros asociados al descarte innecesario de piezas plenamente funcionales. A partir de esta situación de partida, se ha modelado un escenario de mejora deliberadamente conservador.

En primer lugar, se plantea una reducción del 25 % en el volumen de defectos no detectados:

1.600.000 € × 0,25 = 400.000 €

Por otro lado, se estima una reducción del 20 % en el descarte de piezas válidas:

490.000 € × 0,20 = 98.000 €

Estas mejoras generan un ahorro total estimado de 498.000 € mensuales. Para el cálculo de costes, se consideran los recursos humanos necesarios y la infraestructura tecnológica asociada. El equipo técnico estará compuesto por dos especialistas, con un coste conjunto de 10.000 € mensuales, a lo que se suma el coste de la infraestructura cloud distribuida, estimado en 1.200 € al mes para soportar el procesamiento del histórico y los reentrenamientos periódicos.

Con un coste total mensual de 11.200 € y un beneficio estimado de 498.000 €, el proyecto arroja un ROI mensual aproximado del 4.346 %, lo que permite amortizar la inversión tecnológica en menos de un día de operación.

Finalmente, para garantizar la viabilidad ética y legal, se han establecido una serie de principios preventivos desde la fase de diseño. Aunque no se procesan datos personales de clientes finales, el sistema maneja información asociada a líneas de producción y operarios. Por este motivo, todos los identificadores serán sometidos a seudonimización, y el uso del modelo quedará estrictamente limitado a la mejora del proceso de calidad, excluyendo cualquier aplicación disciplinaria o evaluativa sobre el personal. Se priorizará la paridad en tasas de error entre líneas de producción. Tras consultar la literatura especializada, se ha determinado que la métrica de equidad prioritaria será la igualdad de odds (equalized odds), que garantiza que tanto la tasa de falsos positivos como la de falsos negativos sea consistente independientemente de la línea de producción analizada, evitando que el modelo penalice sistemáticamente a unas líneas frente a otras.

### Tabla de métricas y objetivos

| Concepto                    | Situación actual  | Objetivo proyectado | Mejora esperada |
| --------------------------- | ----------------- | ------------------- | --------------- |
| Defectos no detectados      | 8.000 piezas/mes  | 5.000 piezas/mes    | -25 %           |
| Piezas válidas descartadas  | 49.000 piezas/mes | 39.200 piezas/mes   | -20 %           |
| Recall (detección defectos) | 60 %              | 75 %                | +15 pp          |
| Falsos positivos            | 5 %               | 4 %                 | -1 pp           |
| Pérdidas económicas         | 2.090.000 € / mes | 1.592.000 € / mes   | -498.000 €      |


---

## Planificación y recursos

La hoja de ruta operativa traduce las promesas económicas expuestas en el apartado anterior en umbrales técnicos concretos, que actuarán como garantes del proyecto. Para materializar el ROI estimado, el modelo predictivo debe cumplir una serie de objetivos matemáticos estrictos, directamente derivados de la situación actual del sistema de control de calidad.

En primer lugar, en relación con la detección de componentes defectuosos, partimos de una tasa de detección base (recall) del 60 %, lo que implica que actualmente solo 12.000 defectos son identificados correctamente de un total estimado de 20.000. Para alcanzar el ahorro mensual proyectado de 400.000 euros (reducción del 25 % en defectos no detectados), el nuevo modelo deberá elevar esta tasa de detección hasta al menos el 75 %, incrementando de forma significativa la capacidad del sistema para interceptar fallos reales antes de su salida al mercado.

De manera simultánea, para reducir los costes asociados al descarte innecesario de piezas válidas, es imprescindible actuar sobre la tasa de falsos positivos del sistema. Dado que la situación actual presenta una tasa de rechazo erróneo del 5 %, y que el compromiso económico asumido implica una reducción del 20 % de este desperdicio, el objetivo técnico innegociable será mantener la tasa de falsos positivos por debajo del 4 %.

Solo el cumplimiento simultáneo de ambos umbrales técnicos garantiza la coherencia de las estimaciones económicas presentadas y justifica la inversión propuesta.

La ejecución técnica para alcanzar estos objetivos recaerá sobre un equipo multidisciplinar de dos especialistas, que unificarán los roles de ingeniería de datos y ciencia de datos bajo un enfoque end-to-end. Este modelo operativo resulta necesario para orquestar el ciclo completo del sistema, desde la gestión del repositorio de datos industriales —que combina tablas de contexto histórico con flujos de eventos procedentes de sensores y sistemas de inspección, almacenados en ficheros estructurados y particionados temporalmente— hasta la puesta en producción del modelo.

Este ciclo incluye la experimentación iterativa, el enriquecimiento de características relevantes (feature engineering), el entrenamiento de los algoritmos y su posterior despliegue en un entorno productivo con monitorización continua.

En cuanto a la arquitectura tecnológica, el flujo de datos y procesamiento se centraliza en una plataforma distribuida basada en Databricks. El procesamiento masivo y el entrenamiento de los modelos se delegarán en la librería Spark MLlib, aprovechando su capacidad de paralelización para trabajar con el volumen histórico completo. La gobernanza del ciclo de vida de los modelos quedará asegurada mediante MLflow, herramienta encargada del registro de experimentos, control de versiones y trazabilidad de resultados.

Finalmente, la planificación temporal del proyecto se estructura en cuatro hitos críticos alineados con el calendario oficial de la asignatura: el cierre definitivo del alcance para el 27 de febrero de 2026, la finalización de la fase de ingeniería de datos el 20 de marzo de 2026, la entrega del modelo validado el 17 de abril de 2026 y el despliegue final con monitorización activa para el 1 de mayo de 2026.

### Cronograma de hitos

| Hito                             | Fecha       | Descripción                                                       |
| -------------------------------- | ----------- | ----------------------------------------------------------------- |
| Cierre de alcance                | 27 feb 2026 | Confirmación de objetivos, datos y límites del proyecto           |
| Finalización ingeniería de datos | 20 mar 2026 | Limpieza, integración y preprocesamiento de datos                 |
| Modelo validado                  | 17 abr 2026 | Entrenamiento, testeo y validación del modelo predictivo          |
| Despliegue y monitorización      | 1 may 2026  | Producción del sistema, monitorización continua y ajustes finales |
