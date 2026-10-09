import json
import sys

ruta = "evaluation/results/baseline_respuestas.json"
ids = sys.argv[1:] or ["7", "9", "10"]

datos = json.load(open(ruta, encoding="utf-8"))
datos = datos.get("answers", datos) if isinstance(datos, dict) else datos

if isinstance(datos, dict):
    items = [(str(k), v) for k, v in datos.items()]
else:
    items = [(str(v.get("id", i + 1)) if isinstance(v, dict) else str(i + 1), v) for i, v in enumerate(datos)]

primero = items[0][1]
print("Campos de cada entrada:", list(primero.keys()) if isinstance(primero, dict) else type(primero))

for clave, v in items:
    if clave not in ids:
        continue
    print("\n" + "=" * 70)
    print("PREGUNTA", clave)
    for campo, valor in v.items():
        if isinstance(valor, list):
            print(f"\n[{campo}] ({len(valor)} elementos)")
            for x in valor:
                print(" -", str(x)[:220].replace("\n", " "))
        else:
            print(f"[{campo}]", str(valor)[:300].replace("\n", " "))