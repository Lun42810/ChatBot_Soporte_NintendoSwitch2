"""Evalúa el RAG con Ragas (sin el ejecutor interno de Ragas, que falla con nest_asyncio en Python 3.12+).

Ejemplos:
  python evaluation/run_ragas.py --label baseline
  python evaluation/run_ragas.py --label iter1 --languages es --only-generate
  python evaluation/run_ragas.py --label iter1 --languages es --concurrency 1 --ids 1-8
"""
import argparse
import asyncio
import json
import os
import sys
import time
import unicodedata
import re
import unittest.mock as mock
from pathlib import Path
import time

# nest_asyncio rompe asyncio.wait_for/timeout en Python 3.12+ ("Timeout should be used inside a task").
# Este script usa su propio asyncio.run, así que no lo necesita: se sustituye por un módulo vacío ANTES de importar ragas.
import types

_stub = types.ModuleType("nest_asyncio")
_stub.apply = lambda *args, **kwargs: None
sys.modules["nest_asyncio"] = _stub

# Parche del Colab del profe: algunas versiones de ragas fallan al importar VertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", mock.MagicMock())
sys.modules.setdefault("langchain_community.llms.vertexai", mock.MagicMock())

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402

from rag import NO_ANSWER, RAGConfig, answer_question  # noqa: E402

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def is_abstention(text: str) -> bool:
    return _norm(NO_ANSWER) in _norm(text)


def parse_ids(spec: str) -> set:
    """'1-8' -> {1..8}; '1,3,5-7' -> {1,3,5,6,7}; vacío -> conjunto vacío (todas)."""
    ids = set()
    for part in filter(None, (x.strip() for x in (spec or "").split(","))):
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def asyncio_sanity() -> bool:
    """Comprueba que asyncio.wait_for funciona dentro de asyncio.run (sin llamar a ningún LLM)."""
    async def probe():
        return await asyncio.wait_for(asyncio.sleep(0, "ok"), timeout=5)

    try:
        return asyncio.run(probe()) == "ok"
    except RuntimeError as error:
        print(f"[!] asyncio no funciona en este entorno: {error}")
        print(f"    Python {sys.version.split()[0]} | nest_asyncio real cargado: "
              f"{hasattr(sys.modules.get('nest_asyncio'), '__file__')}")
        return False


def build_metrics():
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    return {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(strictness=1),  # strictness=1: Groq no admite n>1
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }


def _is_daily_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return "per day" in msg or "(tpd)" in msg or "requests per day" in msg


async def score_all(rows, metrics, concurrency, retries, cache, save_cache):
    from ragas.dataset_schema import SingleTurnSample

    sem = asyncio.Semaphore(concurrency)
    state = {"daily_limit": False, "too_large": False, "printed": False}

    async def one(row, name, metric):
        key = f"{row['id']}:{name}"
        if key in cache:
            return cache[key]
        sample = SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        )
        async with sem:
            for attempt in range(1, retries + 1):
                if state["daily_limit"] or state["too_large"]:
                    return float("nan")
                try:
                    if hasattr(metric, "_single_turn_ascore"):
                        value = float(await metric._single_turn_ascore(sample, []))
                    else:
                        value = float(await metric.single_turn_ascore(sample))
                    if value == value:                 # no guardar NaN en la caché
                        cache[key] = value
                        save_cache()
                    return value
                except Exception as error:  # límites de Groq, JSON mal formado del juez, etc.
                    print(f"  [aviso] {name} pregunta {row['id']} intento {attempt}/{retries}: "
                          f"{type(error).__name__}: {str(error)[:120]}")
                    if "request too large" in str(error).lower():
                        # Error permanente: la solicitud supera el límite por petición del modelo; reintentar no sirve.
                        state["too_large"] = True
                        if not state["printed"]:
                            state["printed"] = True
                            print("\n[!] MENSAJE COMPLETO DE GROQ:\n" + str(error)[:900] + "\n")
                        return float("nan")
                    if _is_daily_limit(error):
                        state["daily_limit"] = True
                        return float("nan")
                    await asyncio.sleep(5 * attempt)
        return float("nan")

    jobs = [(r["id"], n, one(r, n, m)) for r in rows for n, m in metrics.items()]
    values = await asyncio.gather(*(j[2] for j in jobs))
    out = {}
    for (qid, name, _), value in zip(jobs, values):
        out.setdefault(qid, {})[name] = value
    return out, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="nombre de la corrida (baseline, iter1, ...)")
    p.add_argument("--questions", default=str(ROOT / "evaluation" / "eval_questions.json"))
    p.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-score", type=float)
    p.add_argument("--languages", help='idiomas a indexar, p. ej. "es"; vacío = todos (por defecto usa LANGUAGES del entorno)')
    p.add_argument("--header", action="store_true", help="indexar con el nombre del manual al inicio de cada fragmento")
    p.add_argument("--sleep", type=float, default=2.0, help="pausa entre preguntas (límites de Groq)")
    p.add_argument("--concurrency", type=int, default=2, help="llamadas simultáneas al juez")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--reuse", action="store_true", help="compatibilidad: las respuestas guardadas siempre se reutilizan")
    p.add_argument("--ids", default="", help='evaluar solo estas preguntas, p. ej. "1-8" o "1,3,9-12" (el resto se conserva de la caché)')
    p.add_argument("--fresh", action="store_true", help="ignora respuestas y puntajes guardados de esta etiqueta")
    p.add_argument("--only-generate", action="store_true", help="solo genera y guarda las respuestas (no llama al juez)")
    args = p.parse_args()

    base = RAGConfig.from_env()
    cfg = RAGConfig(
        chunk_size=args.chunk_size or base.chunk_size,
        chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else base.chunk_overlap,
        top_k=args.top_k or base.top_k,
        min_score=args.min_score if args.min_score is not None else base.min_score,
        embedding_model=base.embedding_model,
        languages=args.languages.strip().lower() if args.languages is not None else base.languages,
        header=args.header or base.header,
    )

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    missing = [i["id"] for i in items if not i["ground_truth"].strip()]
    if missing:
        sys.exit(f"Faltan ground_truth en las preguntas {missing}.")

    print(f"[{args.label}] {cfg}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers_path = out / f"{args.label}_respuestas.json"
    scores_path = out / f"{args.label}_puntajes.json"
    cfg_dict = {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "top_k": cfg.top_k,
                "min_score": cfg.min_score, "embedding_model": cfg.embedding_model,
                "languages": cfg.languages, "header": cfg.header}

    answers, scores_cache = {}, {}
    if not args.fresh and answers_path.exists():
        saved = json.loads(answers_path.read_text(encoding="utf-8"))
        if isinstance(saved, list):   # formato anterior: lista de filas con id, response, retrieved_contexts...
            answers = {str(r["id"]): {"response": r["response"], "contexts": r.get("retrieved_contexts", []),
                                      "sources": r.get("retrieved_sources", "")} for r in saved}
        else:
            # Las corridas guardadas antes de existir 'languages' y 'header' equivalen a languages="" y header=False.
            saved_cfg = {"languages": "", "header": False, **saved.get("config", {})}
            if saved_cfg != cfg_dict:
                sys.exit(f"La etiqueta '{args.label}' ya tiene respuestas con otra configuración. "
                         "Usa otra --label o añade --fresh.")
            answers = saved["answers"]
        if scores_path.exists():
            scores_cache = json.loads(scores_path.read_text(encoding="utf-8"))
        print(f"Reanudando: {len(answers)} respuestas y {len(scores_cache)} puntajes guardados.")

    def save_answers():
        answers_path.write_text(json.dumps({"config": cfg_dict, "answers": answers}, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    def save_scores():
        scores_path.write_text(json.dumps(scores_cache, indent=1), encoding="utf-8")

    rows = []
    for item in items:
        key = str(item["id"])
        if key not in answers:
            try:
                result = answer_question(item["question"], config=cfg)
            except Exception as error:
                save_answers()
                sys.exit(f"\nSe detuvo en la pregunta {item['id']}: {type(error).__name__}: {str(error)[:300]}\n"
                         "Lo ya generado quedó guardado. Si es un límite de Groq (429), vuelve a ejecutar el mismo "
                         "comando más tarde y continuará desde aquí.")
            answers[key] = {
                "response": result["answer"],
                "contexts": result["contexts"],
                "sources": "; ".join(f"{s['source']} p.{s['page']}" for s in result["sources"]),
            }
            save_answers()
            time.sleep(args.sleep)
        a = answers[key]
        rows.append({
            "id": item["id"],
            "in_scope": item["in_scope"],
            "user_input": item["question"],
            "response": a["response"],
            "retrieved_contexts": a["contexts"],
            "retrieved_sources": a["sources"],
            "reference": item["ground_truth"],
        })
        print(f"  pregunta {item['id']}: {len(a['contexts'])} fragmentos | abstiene={is_abstention(a['response'])}")

    if args.only_generate:
        print(f"Respuestas guardadas en {answers_path}. Ejecuta de nuevo sin --only-generate para evaluar.")
        return

    # --- Ragas (cada métrica se llama directamente dentro de un asyncio.run propio)
    from langchain_groq import ChatGroq
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    from rag import _get_embeddings

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    print(f"Generador: {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')} | Juez: {judge_model}")
    judge = ChatGroq(
        model=judge_model,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    llm = LangchainLLMWrapper(judge)
    emb = LangchainEmbeddingsWrapper(_get_embeddings(cfg.embedding_model))
    run_config = RunConfig(timeout=240, max_retries=3, max_workers=args.concurrency)

    metrics = build_metrics()
    for metric in metrics.values():
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = emb
        metric.init(run_config)

    if not asyncio_sanity():
        sys.exit("Se detuvo antes de gastar tokens. Prueba con un entorno de Python 3.11 o 3.12.")
    print("Evaluando con Ragas (puede tardar varios minutos)...")
    wanted = parse_ids(args.ids)
    rows_to_score = [r for r in rows if r["id"] in wanted] if wanted else rows
    print(f"Preguntas a puntuar en esta ejecución: {[r['id'] for r in rows_to_score]}")
    _, state = asyncio.run(
        score_all(rows_to_score, metrics, args.concurrency, args.retries, scores_cache, save_scores))
    daily_limit = state["daily_limit"]
    if state["too_large"]:
        sys.exit(f"\nSe detuvo: una solicitud supera el límite por petición (TPM) del modelo juez '{judge_model}'.\n"
                 "Los puntajes ya calculados quedaron guardados. Usa otro juez con un límite mayor "
                 "(RAGAS_JUDGE_MODEL en el .env o en la terminal) y vuelve a ejecutar el mismo comando.")

    detail = pd.DataFrame({
        "id": [r["id"] for r in rows],
        "in_scope": [r["in_scope"] for r in rows],
        "question": [r["user_input"] for r in rows],
        "response": [r["response"] for r in rows],
        "retrieved_sources": [r["retrieved_sources"] for r in rows],
        **{m: [scores_cache.get(f"{r['id']}:{m}", float("nan")) for r in rows] for m in METRIC_COLUMNS},
    })
    detail.to_csv(out / f"{args.label}_detalle.csv", index=False, encoding="utf-8-sig")

    if daily_limit or detail[METRIC_COLUMNS].isna().any().any():
        print("\n[!] Evaluación INCOMPLETA: hay puntajes sin calcular"
              + (" (límite diario de tokens de Groq alcanzado)" if daily_limit else "")
              + ". Vuelve a ejecutar el mismo comando más tarde: solo calculará lo que falta.")
    if detail[METRIC_COLUMNS].isna().all().all():
        sys.exit("Todas las métricas salieron NaN: revisa los avisos de arriba (API key, límites de Groq, versión de ragas).")

    in_scope = detail[detail["in_scope"]]
    out_scope = detail[~detail["in_scope"]]

    def clean(d):
        return {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in d.items()}

    summary = {
        "label": args.label,
        "config": cfg_dict,
        "judge_model": judge_model,
        "completo": not bool(detail[METRIC_COLUMNS].isna().any().any()),
        "ragas_all": clean(detail[METRIC_COLUMNS].mean()),
        "ragas_in_scope": clean(in_scope[METRIC_COLUMNS].mean()),
        "nan_count": {m: int(detail[m].isna().sum()) for m in METRIC_COLUMNS},
        # Control de alucinaciones: ¿se abstiene cuando la respuesta no está en el corpus?
        "abstention_rate_out_of_scope": round(float(out_scope["response"].map(is_abstention).mean()), 3),
        "false_abstention_rate_in_scope": round(float(in_scope["response"].map(is_abstention).mean()), 3),
    }
    (out / f"{args.label}_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
