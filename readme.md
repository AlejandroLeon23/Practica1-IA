
# Clasificación con árbol de decisión: Wine Dataset

Esta práctica entrena árboles de decisión para clasificar las 178 muestras del conjunto `<span>Wine</span>` de scikit-learn. Cada muestra tiene 13 mediciones numéricas de laboratorio y una etiqueta de clase (0, 1 o 2).

## Requisitos y ejecución

* Python 3.9 o posterior.
* `<span>scikit-learn</span>` (versión utilizada para los resultados: 1.8.0).

```
python -m pip install scikit-learn
python app.py
```

El script divide los datos en 142 muestras de entrenamiento (80 %) y 36 de prueba (20 %), con `<span>random_state=42</span>` y distribución estratificada de las clases. Entrena profundidades máximas de 1 a 6 y una variante `<span>None</span>`. Imprime precisión en ambos conjuntos y las reglas de profundidad 2 y sin límite.

## Resultados observados

| `<span>max_depth</span>` | Profundidad real | Hojas | Precisión entrenamiento | Precisión prueba |
| -------------------------- | ---------------- | ----- | ------------------------ | ----------------- |
| 1                          | 1                | 2     | 66.20 %                  | 58.33 %           |
| 2                          | 2                | 4     | 93.66 %                  | 86.11 %           |
| 3                          | 3                | 7     | 99.30 %                  | 94.44 %           |
| 4                          | 4                | 8     | 100.00 %                 | 94.44 %           |
| 5                          | 4                | 8     | 100.00 %                 | 94.44 %           |
| 6                          | 4                | 8     | 100.00 %                 | 94.44 %           |
| `<span>None</span>`      | 4                | 8     | 100.00 %                 | 94.44 %           |

La máxima precisión **observada en esta prueba** fue 94.44 % (34 aciertos de 36), alcanzada desde `<span>max_depth=3</span>`. El árbol sin límite creció hasta profundidad real 4; por eso pedir máximos 5 y 6 no cambió sus reglas ni sus resultados. Los árboles más profundos separaron completamente el entrenamiento, pero no elevaron la precisión de prueba respecto al de profundidad 3. Este patrón sugiere que añadir reglas ya no mejoró la generalización en esta partición; una sola prueba pequeña no demuestra que la profundidad 3 sea universalmente óptima.

## Interpretación de las reglas

El árbol de profundidad 2 divide primero según `<span>color_intensity</span>` (umbral 3.82). En una de sus ramas usa `<span>ash</span>` y en la otra `<span>flavanoids</span>`. Cada camino termina con una clase predicha. Al retirar el límite aparecen divisiones adicionales con `<span>od280/od315_of_diluted_wines</span>`, `<span>alcalinity_of_ash</span>`, `<span>proline</span>` y `<span>malic_acid</span>`. Esas condiciones permiten distinguir muestras que caían juntas en el árbol corto, a costa de más complejidad.

## ¿Es adecuado el conjunto de datos?

Sí, para una práctica de  **clasificación supervisada** : hay una etiqueta conocida por muestra, 13 atributos numéricos y tres clases. Permite calcular precisión y leer reglas sencillas. Su tamaño reducido facilita aprender, aunque limita cuán sólidas son las conclusiones fuera de este conjunto.

Las variables que aparecen en las reglas del modelo, especialmente `<span>color_intensity</span>` y `<span>flavanoids</span>`, son útiles para esta partición. Que el árbol use una variable no demuestra que cause la clase ni que sea siempre la más importante. Para un estudio más amplio serían valiosos metadatos sobre las condiciones de medición y procedencia de las muestras, además de nuevas mediciones de laboratorio comparables; habría que comprobar su calidad y si aportan información real.

**Nota metodológica:** comparar profundidades mediante la misma prueba es útil para explorar, pero seleccionar la profundidad basándose en esa prueba sesgaría la estimación final. En un proyecto formal se elegiría la profundidad con validación cruzada sobre entrenamiento y se usaría la prueba solo una vez al final.

## Archivos

* `<span>app.py</span>`: entrenamiento, tabla de resultados y reglas.
* `<span>README.md</span>`: procedimiento, resultados y análisis.
