# Proyecto-Tratamiento-de-Datos-AMJ-y-JMIG

# **Análisis de Textos y Regresión con Modelos Transformer**

## Autor
- **Alejandro Mesquida Jiménez, 100429586**
- **José María Iranzo González, 100406233**

## Descripción General
Este proyecto se centra en el análisis y la predicción de datos utilizando diferentes métodos de representación vectorial y técnicas de modelado para problemas de regresión. Se implementaron diversas técnicas, desde el preprocesamiento de textos hasta el uso de modelos transformer para la predicción de resultados.

### Metodología Utilizada
1. **Análisis de variables de entrada**
   - Visualización a través de histogramas de la relación entre la variable `rating` y las categorías en la variable `categories`. Se ha tenido en cuenta el numero de repeticiones de categorias para despreciar las que menos apariciones tiene. Como categorias más influyentes tenemos bon_appetit y las diferentes categorias de intolerancias o minorias gastronómicas. 
   - Estos resultados tienen un sentido, en primer lugar que bon_appetit significa buen provecho en frances, lo que augura que el plato con dicha categoría tiene un buen gusto. En segundo lugar, las personas con alergias, intolerancias o que pertenecen a minorías gastronómicas valoran mejor aquellos platos que están hechos a su medida.  

2. **Implementación de un pipeline para el preprocesado de textos**
   - Uso de librerías como NLTK y SpaCy para el tokenizado, eliminación de stopwords y lematización.
   - Explicación de cómo se manejaron los textos sin preprocesar para la entrada en modelos basados en transformers.
   - Resultados del preprocesamiento y cómo afectaron las representaciones vectoriales generadas.

3. **Representación vectorial de los documentos**
   - **TF-IDF**: Implementación y resultados obtenidos con este método. Discusión sobre la importancia de las palabras clave en los documentos.
   - **Word2Vec**: Descripción de cómo se promedió los embeddings de las palabras para formar vectores de documento. Análisis de similitud entre documentos y palabras clave.
   - **Embeddings contextuales (BERT)**: Explicación del proceso de obtención de embeddings BERT y cómo se utilizaron para mejorar las representaciones de documentos. Comparación con otros métodos.

4. **Entrenamiento y evaluación de modelos de regresión**
   - Uso de redes neuronales en PyTorch para el aprendizaje automático. En concreto una red neuronal multicapa con normalización y ReLu. El modelo de vectorización se le pedirá al usuario por teclado, para facilitar su uso. 
   - Implementación de *Random Forest* en Scikit-learn para comparación con el modelo de red neuronal. Se obtienen diferentes resultados. En este caso será el usuario el que deberá cambiar los parámetros, ya que hay implementado un optimizador de hiperparámetros, que escoje los que mejor &R^2& proporciona.
   - Evaluación con métricas como Loss y &R^2& para determinar la efectividad de los modelos. Así como gráficas para facilitar la comprensión. Se aprecia en los resultados diferentes aspectos, por un lado, que al reducir la base de datos para hacerla más manejable, estamos reduciendo las posibilidades de que nuestro modelo mejore. Esto se ve reflejado en las métricas que tienen peores valores. Sin embargo nos damos cuenta de que BERT consigue mucho mejor resultado que los otros dos modelos de vectorización. Tanto para la red neuronal, como para el Random Forest. 

5. **Comparación de lo obtenido con el fine-tuning de un modelo preentrenado**
   - Uso de Hugging Face para ajustar un modelo transformer con una cabeza de regresión.
   - Comparación de los resultados con las técnicas anteriores y discusión sobre las mejoras obtenidas.

### Extra
- **Summarizer**: Implementación de un summarizer para reducir la longitud de los textos de la variable `description`. Hemos tomado una porción pequeña del dataset para poder demostrar su uso. 
- **Clustering de diferentes variables**: Uso de un algoritmo k-means para agrupar diferentes variables. Análisis de los clusters resultantes y la relevancia de estos para el problema de estudio.
