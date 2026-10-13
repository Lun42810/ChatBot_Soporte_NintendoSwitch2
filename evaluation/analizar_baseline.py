"""Analiza un baseline ya evaluado: ¿el manual correcto llegó entre los fragmentos recuperados?

Uso (desde la raíz del proyecto):
  python evaluation/analizar_baseline.py baseline
No llama a ninguna API: solo lee evaluation/results/<label>_detalle.csv y eval_questions.json.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "evaluation" / "results"
NO_ANSWER = "No encontré información sobre esto en la base de conocimientos."
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def norm(text):
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text))


label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
detail = pd.read_csv(RES / f"{label}_detalle.csv", encoding="utf-8-sig")
questions = {q["id"]: q for q in json.loads((ROOT / "evaluation" / "eval_questions.json").read_text(encoding="utf-8"))}

rows = []
for _, r in detail.iterrows():
    q = questions[int(r["id"])]
    if not q["in_scope"]:
        continue
    hint = q["source_hint"]
    raw = r["retrieved_sources"]
    # Si el modelo se abstiene, rag.py no devuelve fuentes: la celda queda vacía (NaN) y NO significa que no se recuperó nada.
    sources = [] if pd.isna(raw) else [x.strip() for x in str(raw).split(";") if x.strip()]
    docs = [re.sub(r"\s+p\.\S*$", "", s) for s in sources]
    hits = sum(1 for d in docs if d == hint)
    rows.append({
        "id": int(r["id"]),
        "esperado": hint.replace("NSwitch2_", "").replace("_EUR.pdf", ""),
        "fragmentos_del_manual_correcto": f"{hits}/{len(docs)}" if docs else "sin dato",
        "llego": (hits > 0) if docs else None,
        "contexto_cubria": "si" if float(r["context_recall"]) >= 0.5 else "no",
        "se_abstuvo": norm(NO_ANSWER) in norm(r["response"]),
        **{m: round(float(r[m]), 2) for m in METRICS},
        "otros_manuales": ", ".join(sorted({d.replace('NSwitch2_', '').replace('_EUR.pdf', '') for d in docs if d != hint})),
    })

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60)
print("\n=== Preguntas del corpus: ¿llegó el manual correcto? ===")
print(out.drop(columns=["otros_manuales"]).to_string(index=False))

print("\n=== Preguntas con abstención falsa ===")
print("(Sin dato: al abstenerse no se guardan las fuentes. 'contexto_cubria' usa context_recall >= 0.5 como pista de si el contexto traía la respuesta.)")
print(out[out["se_abstuvo"]][["id", "esperado", "fragmentos_del_manual_correcto", "contexto_cubria", "context_precision", "context_recall"]].to_string(index=False))

print("\n=== Resumen ===")
con_dato = out[out["llego"].notna()]
print(f"Preguntas con fuentes guardadas: {len(con_dato)}/{len(out)}; en {int(con_dato['llego'].sum())} llegó al menos un fragmento del manual correcto.")
print(f"Preguntas sin fuentes guardadas (abstenciones): {int(out['llego'].isna().sum())}")
for flag, name in [(True, "llegó el manual correcto"), (False, "NO llegó el manual correcto")]:
    sub = out[out["llego"] == flag]
    if len(sub):
        print(f"\nCuando {name} (n={len(sub)}): medias")
        print(sub[METRICS].mean().round(3).to_string())
sin = out[out["llego"].isna()]
if len(sin):
    print(f"\nSin fuentes guardadas, n={len(sin)}: medias")
    print(sin[METRICS].mean().round(3).to_string())
print("\nNota: este análisis cuenta archivos, no idiomas. Un fragmento del manual correcto pero en otro idioma cuenta como acierto.")
low = out.sort_values("answer_relevancy").head(4)[["id", "esperado", "llego", "se_abstuvo", "contexto_cubria", "answer_relevancy"]]
print("\n=== Las 4 con peor answer_relevancy ===")
print(low.to_string(index=False))

out.to_csv(RES / f"{label}_analisis.csv", index=False, encoding="utf-8-sig")
print(f"\nGuardado en {RES / (label + '_analisis.csv')}")