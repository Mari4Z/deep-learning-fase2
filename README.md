# Proyecto integrador de Deep Learning, Fase 2: replicación, traslado y transferencia de aprendizaje

Universidad Externado de Colombia · Deep Learning
Autoras: María José Cortés Romero y María Isabell Guzmán Faneyte

Este repositorio contiene el código y los resultados del informe de la Segunda entrega. Se replica el método de
Benítez-Andrades et al. (2022) para detectar racismo y xenofobia en tuits en español (CNN, LSTM, CNN+LSTM, BETO y mBERT),
se traslada a un corpus independiente (*Contextualized Hate Speech*, medios argentinos) con transferencia de aprendizaje y se
compara BETO contra líneas base clásicas (TF-IDF con regresión logística y SVM lineal) sobre un conjunto de prueba congelado.

## Contenido

| Archivo o carpeta | Qué contiene |
|---|---|
| `Fase_2_completo_7_1_a_7_4.ipynb` | Cuaderno ejecutado con las secciones 7.1 (replicación), 7.2 (traslado al problema local), 7.3 (estrategia de transferencia) y 7.4 (dinámica del entrenamiento y comparación con la línea base) |
| `bert_worker.py`, `bert_worker_local.py`, `bert_worker_transfer.py`, `bert_worker_final.py` | Procesos de entrenamiento de BERT de cada sección; el cuaderno los invoca (cada corrida se entrena en un proceso aparte) |
| `salidas_71/*.csv`, `salidas_71/*.png` | Tablas y figuras que genera el cuaderno |
| `salidas_71/corridas_*` | Resultados de cada corrida (historial, logits de validación y de prueba, normas del gradiente por paso) |
| `salidas_71/particion_local.csv` | Partición del corpus local (identificador y partición); el conjunto de prueba está congelado |
| `salidas_71/configuracion_elegida.json` | Configuración elegida en 7.3 con la regla fijada de antemano |
| `requirements.txt` | Versiones de las librerías |

## Cómo reproducirlo

1. Crear un entorno con Python 3.12 e instalar `pip install -r requirements.txt`.
2. Abrir `Fase_2_completo_7_1_a_7_4.ipynb` y ejecutar las celdas en orden, con el directorio de trabajo en la raíz del repositorio.
3. Las corridas de BERT ya ejecutadas están en `salidas_71/corridas_*`; si existen, el cuaderno **no vuelve a entrenarlas** y reconstruye las tablas y figuras desde sus resultados. Para entrenar desde cero hay que borrar esas carpetas (varias horas: unas 8 en total en un Mac con chip Apple M4 y 16 GB).
4. El cuaderno descarga los datos (no se incluyen en el repositorio): el corpus y el código del artículo desde Zenodo, los embeddings preentrenados desde Zenodo (lectura en *streaming* de un archivo de 3.4 GB; solo se conserva el subconjunto del vocabulario) y el corpus local desde Hugging Face con una versión fija.

Todas las semillas están fijadas. Las corridas de BERT en GPU no son idénticas bit a bit entre ejecuciones; por eso cada configuración final se repite con varias semillas y se reporta la variabilidad.

## Datos y modelos de terceros

- Corpus y código del artículo: González-Jiménez y Benítez-Andrades (2022), <https://doi.org/10.5281/zenodo.5886624>.
- Embeddings preentrenados: Cañete (2019), <https://doi.org/10.5281/zenodo.3234051>.
- Corpus local: Pérez et al. (2023), <https://huggingface.co/datasets/piuba-bigdata/contextualized_hate_speech> (el repositorio no declara licencia: se usa con fines académicos).
- Modelos preentrenados: BETO (`dccuchile/bert-base-spanish-wwm-uncased` y `-cased`; Cañete et al., 2020) y mBERT (`bert-base-multilingual-cased`; Devlin et al., 2019).

## Declaración de uso de IA generativa

Se utilizó Claude (Anthropic) como apoyo en la revisión del código, la ejecución de los experimentos y la redacción. Las autoras revisaron los resultados y pueden sustentar cada decisión técnica.
