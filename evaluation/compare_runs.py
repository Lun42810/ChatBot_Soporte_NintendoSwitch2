"""Compara dos corridas ya evaluadas: python evaluation/compare_runs.py baseline iter1

Genera evaluation/results/comparacion_<a>_vs_<b>.md (y .png si hay matplotlib) con:
  - métricas Ragas solo sobre las preguntas del corpus (la comparación principal),
  - métricas Ragas sobre las 20 preguntas,
  - control de alucinaciones (abstenciones).
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
a, b = (json.loads((RES / f"{n}_resumen.json").read_text(encoding="utf-8")) for n in sys.argv[1:3])

# --- Avisos de validez
for s in (a, b):
    if not s.get("completo", True):
        print(f"[!] La corrida '{s['label']}' está INCOMPLETA (hay NaN): sus medias no son comparables.")
if a.get("judge_model") != b.get("judge_model"):
    print(f"[!] Jueces distintos: {a.get('judge_model')} vs {b.get('judge_model')}. La comparación no es válida.")

cfg_keys = sorted(set(a["config"]) | set(b["config"]))
defaults = {"languages": "", "header": False}
changes = []
for k in cfg_keys:
    x, y = a["config"].get(k, defaults.get(k)), b["config"].get(k, defaults.get(k))
    if x != y:
        changes.append(f"{k}: {x!r} -> {y!r}")
print("\nCambios de configuración:", "; ".join(changes) if changes else "ninguno (¿la etiqueta nueva usa la misma configuración?)")
if len(changes) > 1:
    print("[!] Hay más de un cambio de configuración: no se podrá atribuir la mejora a un solo parámetro.")


def cell(x):
    return "n/d" if x is None else f"{x:.3f}"


def row(name, x, y):
    delta = "n/d" if x is None or y is None else f"{y - x:+.3f}"
    return f"| {name} | {cell(x)} | {cell(y)} | {delta} |"


def header():
    return [f"| Métrica | {a['label']} | {b['label']} | Δ |", "|---|---|---|---|"]


def metrics_table(section):
    lines = header()
    for k in a[section]:
        lines.append(row(k, a[section].get(k), b[section].get(k)))
    return lines


abst = header()
abst.append(row("abstención fuera del corpus (debe ser alta)", a.get("abstention_rate_out_of_scope"), b.get("abstention_rate_out_of_scope")))
abst.append(row("abstención falsa dentro del corpus (debe ser baja)", a.get("false_abstention_rate_in_scope"), b.get("false_abstention_rate_in_scope")))

md = ["## Solo preguntas del corpus (15)", *metrics_table("ragas_in_scope"),
      "", "## Las 20 preguntas", *metrics_table("ragas_all"),
      "", "## Control de alucinaciones", *abst]
text = "\n".join(md)
print("\n" + text)
name = f"comparacion_{a['label']}_vs_{b['label']}"
(RES / f"{name}.md").write_text(text, encoding="utf-8")

try:
    import matplotlib.pyplot as plt

    keys = list(a["ragas_in_scope"])
    xs = range(len(keys))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.2 for i in xs], [a["ragas_in_scope"][k] or 0 for k in keys], 0.4, label=a["label"])
    plt.bar([i + 0.2 for i in xs], [b["ragas_in_scope"][k] or 0 for k in keys], 0.4, label=b["label"])
    plt.xticks(list(xs), keys, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Métricas Ragas (preguntas del corpus): antes vs. después")
    plt.tight_layout()
    plt.savefig(RES / f"{name}.png", dpi=150)
    print(f"\nGuardado: {name}.md y {name}.png")
except ImportError:
    print(f"\nGuardado: {name}.md (sin gráfico: falta matplotlib)")
