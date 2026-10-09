"""Mide la RECUPERACIÓN sin llamar a Groq (solo embeddings locales): no gasta tokens.

Para cada pregunta con manual esperado (source_hint) comprueba si el manual correcto aparece en el top_k,
en qué posición, y qué parte de los fragmentos recuperados está en español.

Ejemplos:
  python evaluation/retrieval_check.py --label baseline
  python evaluation/retrieval_check.py --label iter1 --languages es --header
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402

from rag import RAGConfig, detect_language, retrieve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--questions", default=str(ROOT / "evaluation" / "eval_questions.json"))
    p.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--top-k", type=int)
    p.add_argument("--languages", help='p. ej. "es" o "es,en"; vacío = todos')
    p.add_argument("--header", action="store_true", help="indexar con el nombre del manual al inicio de cada fragmento")
    p.add_argument("--languages", help='idiomas a indexar, p. ej. "es"; vacío = todos')
    p.add_argument("--header", action="store_true", help="indexar con el nombre del manual en cada fragmento")
    args = p.parse_args()

    base = RAGConfig.from_env()
    cfg = RAGConfig(
        chunk_size=args.chunk_size or base.chunk_size,
        chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else base.chunk_overlap,
        top_k=args.top_k or base.top_k,
        min_score=0.0,
        embedding_model=base.embedding_model,
        languages=args.languages if args.languages is not None else base.languages,
        header=args.header or base.header,
        languages=args.languages.strip().lower() if args.languages is not None else base.languages,
        header=args.header or base.header,
    )
    print(f"[{args.label}] chunk={cfg.chunk_size}/{cfg.chunk_overlap} top_k={cfg.top_k} "
          f"idiomas={cfg.languages or 'todos'} encabezado={cfg.header}")
    print("(la primera vez con una configuración nueva se construye el índice: tarda unos minutos)")

    rows = []
    for item in json.loads(Path(args.questions).read_text(encoding="utf-8")):
        if not item["in_scope"] or not item.get("source_hint", "").endswith(".pdf"):
            continue
        hits = retrieve(item["question"], cfg)
        sources = [Path(d.metadata.get("source", "")).name for d, _ in hits]
        langs = [d.metadata.get("lang") or detect_language(d.page_content) for d, _ in hits]
        rank = next((i + 1 for i, s in enumerate(sources) if s == item["source_hint"]), None)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item["source_hint"],
            "hit": rank is not None,
            "rank": rank,
            "pct_manual_correcto": round(100 * sources.count(item["source_hint"]) / max(1, len(sources))),
            "pct_espanol": round(100 * langs.count("es") / max(1, len(langs))),
            "top1": sources[0] if sources else "",
        })
        print(f"  Q{item['id']:>2} hit={'sí' if rank else 'NO'} rank={rank} "
              f"correcto={rows[-1]['pct_manual_correcto']}% español={rows[-1]['pct_espanol']}%")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"recuperacion_{args.label}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "label": args.label,
        "top_k": cfg.top_k,
        "hit_rate": round(float(df["hit"].mean()), 3),
        "top1_correcto": round(float((df["rank"] == 1).mean()), 3),
        "pct_manual_correcto_medio": round(float(df["pct_manual_correcto"].mean()), 1),
        "pct_espanol_medio": round(float(df["pct_espanol"].mean()), 1),
    }
    (out / f"recuperacion_{args.label}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
