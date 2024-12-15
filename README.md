# Proyecto-Tratamiento-de-Datos-AMJ-y-JMIG

# Nombre del Proyecto: **Análisis de Textos y Regresión con Modelos Transformer**

## Autor
- **Alejandro Mesquida Jiménez, 1004**
- **José María Iranzo González, 100406233**

## Descripción General
Este proyecto se centra en el análisis y la predicción de datos utilizando diferentes métodos de representación vectorial y técnicas de modelado para problemas de regresión. Se implementaron diversas técnicas, desde el preprocesamiento de textos hasta el uso de modelos transformer para la predicción de resultados.

### Metodología Utilizada
1. **Análisis de variables de entrada**
   - Visualización de la relación entre las variables de salida y las categorías en la variable `categories`. Se utilizaron diagramas de dispersión y gráficos de correlación para identificar las relaciones más significativas.
   - Discusión sobre la relevancia de estas relaciones para el problema de estudio y su impacto en la predicción.

2. **Implementación de un pipeline para el preprocesado de textos**
   - Uso de librerías como NLTK y SpaCy para el tokenizado, eliminación de stopwords y lematización.
   - Explicación de cómo se manejaron los textos sin preprocesar para la entrada en modelos basados en transformers.
   - Resultados del preprocesamiento y cómo afectaron las representaciones vectoriales generadas.

3. **Representación vectorial de los documentos**
   - **TF-IDF**: Implementación y resultados obtenidos con este método. Discusión sobre la importancia de las palabras clave en los documentos.
   - **Word2Vec**: Descripción de cómo se promedió los embeddings de las palabras para formar vectores de documento. Análisis de similitud entre documentos y palabras clave.
   - **Embeddings contextuales (BERT)**: Explicación del proceso de obtención de embeddings BERT y cómo se utilizaron para mejorar las representaciones de documentos. Comparación con otros métodos.

4. **Entrenamiento y evaluación de modelos de regresión**
   - Uso de redes neuronales en PyTorch para el aprendizaje automático.
   - Implementación de K-NN, SVM y Random Forest en Scikit-learn para comparación.
   - Evaluación con métricas como RMSE y MAE para determinar la efectividad de los modelos.

5. **Comparación de lo obtenido con el fine-tuning de un modelo preentrenado**
   - Uso de Hugging Face para ajustar un modelo transformer con una cabeza de regresión.
   - Comparación de los resultados con las técnicas anteriores y discusión sobre las mejoras obtenidas.

### Extra
- **Summarizer**: Implementación de un summarizer para reducir la longitud de los textos de la variable `description`. 
- **Clustering de diferentes variables**: Uso de un algoritmo k-means para agrupar diferentes variables. Análisis de los clusters resultantes y la relevancia de estos para el problema de estudio.

## Instrucciones de Instalación
```bash
pip install numpy pandas scikit-learn nltk gensim torch transformers
