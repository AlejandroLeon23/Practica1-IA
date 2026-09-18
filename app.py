"""Práctica de clasificación con árboles de decisión y el dataset Wine."""

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def main():
    wine = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.20,
        random_state=42,
        stratify=wine.target,
    )

    print(f"Muestras: {len(wine.data)} | Variables: {len(wine.feature_names)}")
    print(f"Entrenamiento: {len(x_train)} | Prueba: {len(x_test)}")
    print("Profundidad solicitada | Profundidad real | Hojas | Entrenamiento | Prueba")

    resultados = []
    for profundidad in [1, 2, 3, 4, 5, 6, None]:
        arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
        arbol.fit(x_train, y_train)
        precision_entrenamiento = accuracy_score(y_train, arbol.predict(x_train))
        precision_prueba = accuracy_score(y_test, arbol.predict(x_test))
        resultados.append((profundidad, arbol, precision_entrenamiento, precision_prueba))
        print(
            f"{str(profundidad):>21} | {arbol.get_depth():>16} | "
            f"{arbol.get_n_leaves():>5} | {precision_entrenamiento:>12.2%} | "
            f"{precision_prueba:.2%}"
        )

    for profundidad, arbol, _, _ in resultados:
        if profundidad in (2, None):
            print(f"\nREGLAS: max_depth={profundidad}")
            print(export_text(arbol, feature_names=list(wine.feature_names)))

    maxima = max(resultado[3] for resultado in resultados)
    mejores = [str(profundidad) for profundidad, _, _, p in resultados if p == maxima]
    print(f"Máxima precisión observada en prueba: {maxima:.2%} "
          f"(max_depth={', '.join(mejores)})")
    print("Nota: elegir el modelo usando estos mismos datos de prueba sesga la comparación; "
          "para seleccionar un modelo formalmente se requiere validación separada.")


if __name__ == "__main__":
    main()