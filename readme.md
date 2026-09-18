
# Clasificación con árbol de decisión: Wine Dataset

Esta práctica entrena árboles de decisión para clasificar las 178 muestras del conjunto `Wine` de scikit-learn. Cada muestra tiene 13 mediciones numéricas de laboratorio y una etiqueta de clase (0, 1 o 2).

## Requisitos y ejecución

* Python 3.11 o posterior.
* `scikit-learn` (versión utilizada para los resultados: 1.9.1).

```
python3 -m pip install -r requirements.txt
python3 app.py
```

El script divide los datos en 142 muestras de entrenamiento (80 %) y 36 de prueba (20 %), con `random_state=42` y distribución estratificada de las clases. Entrena profundidades máximas de 1 a 6 y una variante `None`. Imprime precisión en ambos conjuntos y las reglas de todas las profundidades y la importancia de cada variable del árbol sin límite. En esta práctica, «precisión» significa exactitud (`accuracy`): número de predicciones correctas dividido entre el total de muestras.

## Resultados observados

| `max_depth` | Profundidad real | Hojas | Precisión entrenamiento | Precisión prueba |
| -------------------------- | ---------------- | ----- | ------------------------ | ----------------- |
| 1                          | 1                | 2     | 66.20 %                  | 58.33 %           |
| 2                          | 2                | 4     | 93.66 %                  | 86.11 %           |
| 3                          | 3                | 7     | 99.30 %                  | 94.44 %           |
| 4                          | 4                | 8     | 100.00 %                 | 94.44 %           |
| 5                          | 4                | 8     | 100.00 %                 | 94.44 %           |
| 6                          | 4                | 8     | 100.00 %                 | 94.44 %           |
| `None`      | 4                | 8     | 100.00 %                 | 94.44 %           |

La máxima precisión **observada en esta prueba** fue 94.44 % (34 aciertos de 36), alcanzada desde `max_depth=3`. El árbol sin límite creció hasta profundidad real 4; por eso pedir máximos 5 y 6 no cambió sus reglas ni sus resultados. Los árboles más profundos separaron completamente el entrenamiento, pero no elevaron la precisión de prueba respecto al de profundidad 3. Este patrón sugiere que añadir reglas ya no mejoró la generalización en esta partición; una sola prueba pequeña no demuestra que la profundidad 3 sea universalmente óptima.

## Interpretación de las reglas

El árbol de profundidad 2 divide primero según `color_intensity` (umbral 3.82). En una de sus ramas usa `ash` y en la otra `flavanoids`. Cada camino termina con una clase predicha. Al retirar el límite aparecen divisiones adicionales con `od280/od315_of_diluted_wines`, `alcalinity_of_ash`, `proline` y `malic_acid`. Esas condiciones permiten distinguir muestras que caían juntas en el árbol corto, a costa de más complejidad.

## ¿Es adecuado el conjunto de datos?

Sí, para una práctica de  **clasificación supervisada** : hay una etiqueta conocida por muestra, 13 atributos numéricos y tres clases. Permite calcular precisión y leer reglas sencillas. Su tamaño reducido facilita aprender, aunque limita cuán sólidas son las conclusiones fuera de este conjunto.

Las variables que aparecen en las reglas del modelo, especialmente `color_intensity` y `flavanoids`, son útiles para esta partición. Que el árbol use una variable no demuestra que cause la clase ni que sea siempre la más importante. Para un estudio más amplio serían valiosos metadatos sobre las condiciones de medición y procedencia de las muestras, además de nuevas mediciones de laboratorio comparables; habría que comprobar su calidad y si aportan información real.

**Nota metodológica:** comparar profundidades mediante la misma prueba es útil para explorar, pero seleccionar la profundidad basándose en esa prueba sesgaría la estimación final. En un proyecto formal se elegiría la profundidad con validación cruzada sobre entrenamiento y se usaría la prueba solo una vez al final.

## Archivos

* `app.py`: entrenamiento, tabla de resultados y reglas.
* `readme.md`: procedimiento, resultados y análisis.

## Opinión sobre el cambio de profundidad

En mi opinión, el árbol de profundidad 1 es demasiado sencillo porque solo tiene dos hojas para distinguir tres clases. Al aumentar la profundidad aparecen más preguntas y caminos que separan mejor las muestras: la exactitud de prueba pasa de 58.33 % a 94.44 %. El máximo que probé como límite numérico fue 6, pero la profundidad máxima que realmente alcanzó este árbol fue 4. Esto no es un máximo universal de DecisionTreeClassifier: depende de los datos de entrenamiento y de los demás parámetros. En esta ejecución, a profundidad 4 las hojas ya son puras y el algoritmo no necesita seguir dividiendo.

Con `max_depth=None`, el árbol obtiene 100 % en entrenamiento y 94.44 % en prueba. Frente al límite 2 añade condiciones y mejora la prueba; frente al límite 3 añade una división con `malic_acid` y una hoja, pero no mejora los aciertos de prueba. Frente a los límites 4, 5 y 6 no hay diferencia en las reglas. Considero que el árbol de profundidad 3 ofrece un buen equilibrio en esta comparación, aunque hace falta validación independiente para elegirlo formalmente. Aprender perfectamente el entrenamiento no garantiza acertar en muestras nuevas.

## Características fundamentales y posibles mejoras

El dataset sí cumple los requisitos para esta actividad: sus 178 filas tienen una clase conocida, sus 13 características son numéricas y no tiene valores faltantes. Las tres clases cuentan con 59, 71 y 48 muestras, respectivamente; la separación estratificada mantiene aproximadamente esa proporción en entrenamiento y prueba. Un árbol puede establecer umbrales directamente sobre estas mediciones, sin necesidad de normalizarlas. Las etiquetas identifican tres clases de vino, no una puntuación de calidad.

En el árbol sin límite, las características que más contribuyen a reducir la impureza son:

| Característica | Importancia | Por qué ayuda en este árbol |
| --- | --- | --- |
| `flavanoids` | 40.81 % | Su concentración separa las clases en la rama de mayor intensidad de color. |
| `color_intensity` | 40.02 % | Es la primera división y permite separar grandes grupos de muestras. |
| `proline` | 11.10 % | Distingue muestras que comparten intensidad de color alta y mayor concentración de flavonoides. |

Las demás divisiones utilizan cenizas, alcalinidad de cenizas, ácido málico y la relación óptica OD280/OD315. Estas importancias describen este ajuste; no prueban causalidad ni que las variables con importancia cero sean inútiles en otros entrenamientos.

Yo añadiría mediciones de pH y azúcar residual si estuvieran disponibles, porque podrían aportar información química complementaria. Comprobaría su contribución mediante validación sobre entrenamiento antes de incluirlas definitivamente. También reuniría más muestras de diferentes cosechas y registraría las condiciones de laboratorio para evaluar si el modelo funciona fuera de esta pequeña colección. Evitaría introducir identificadores que revelen directamente la clase que se quiere predecir.

## Evidencia de ejecución

`resultados.txt` contiene la salida completa de `python3 app.py`, incluidas todas las reglas. `requirements.txt` fija la versión utilizada de scikit-learn. El dataset viene incluido en esa biblioteca y no requiere descargar un CSV por separado.
