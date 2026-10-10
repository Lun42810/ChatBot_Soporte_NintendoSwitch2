## Solo preguntas del corpus (15)
| Métrica | baseline | iter1 | Δ |
|---|---|---|---|
| faithfulness | 0.728 | 0.729 | +0.001 |
| answer_relevancy | 0.571 | 0.892 | +0.321 |
| context_precision | 0.696 | 0.625 | -0.071 |
| context_recall | 0.722 | 0.750 | +0.028 |

## Las 20 preguntas
| Métrica | baseline | iter1 | Δ |
|---|---|---|---|
| faithfulness | 0.596 | 0.729 | +0.133 |
| answer_relevancy | 0.432 | 0.892 | +0.460 |
| context_precision | 0.522 | 0.625 | +0.103 |
| context_recall | 0.542 | 0.750 | +0.208 |

## Control de alucinaciones
| Métrica | baseline | iter1 | Δ |
|---|---|---|---|
| abstención fuera del corpus (debe ser alta) | 1.000 | 1.000 | +0.000 |
| abstención falsa dentro del corpus (debe ser baja) | 0.200 | 0.067 | -0.133 |