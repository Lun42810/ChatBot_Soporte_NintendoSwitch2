"""Ejecuta las preguntas Ragas y los siete casos de comportamiento sobre el agente."""
import argparse
import asyncio
import json
import os
import sys
import time
import types
from unittest.mock import MagicMock
from pathlib import Path

_nest_stub = types.ModuleType("nest_asyncio")
_nest_stub.apply = lambda *args, **kwargs: None
sys.modules["nest_asyncio"] = _nest_stub
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())
sys.modules.setdefault("langchain_community.llms.vertexai", MagicMock())

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from agent import run_agent
import rag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="1-15", help="IDs del dataset de Avance 2")
    parser.add_argument("--output", default=str(ROOT / "evaluation" / "results" / "agent_eval.json"))
    parser.add_argument("--run-behavior-cases", action="store_true")
    args = parser.parse_args()
    questions = json.loads((ROOT / "evaluation" / "eval_questions.json").read_text(encoding="utf-8"))
    selected = set()
    for part in filter(None, (x.strip() for x in args.ids.split(","))):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    output = {"evaluation_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
              "judge_model": os.getenv("RAGAS_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")),
              "question_ids": sorted(selected), "questions": [], "pipeline_fijo_questions": [],
              "cost": {}, "ragas": {}}
    def checkpoint():
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint()
    rows, costs = [], []
    for row in questions:
        if row["id"] not in selected: continue
        try:
            result = run_agent(row["question"])
        except Exception as error:
            rows.append({"id": row["id"], "user_input": row["question"],
                         "response": "", "retrieved_contexts": [], "reference": row["ground_truth"],
                         "tools": [], "llm_calls": 0, "tokens": 0, "elapsed_seconds": 0,
                         "error": f"{type(error).__name__}: {str(error)[:300]}"})
            print(f"{row['id']}: ERROR {type(error).__name__}: {str(error)[:160]}")
            output["questions"] = rows
            checkpoint()
            continue
        rows.append({"id": row["id"], "user_input": row["question"], "response": result["answer"],
                     "retrieved_contexts": result["contexts"], "reference": row["ground_truth"],
                     "tools": result["tool_calls"], "llm_calls": result["llm_calls"],
                     "tokens": result["tokens"], "elapsed_seconds": result["elapsed_seconds"]})
        costs.append(rows[-1])
        print(f"{row['id']}: {result['llm_calls']} llamadas, {result['tokens']} tokens, {result['elapsed_seconds']} s")
        output["questions"] = rows
        output["cost"]["agent"] = _averages(costs)
        checkpoint()
    fixed_rows, fixed_cost = _fixed_pipeline_cost([row for row in questions if row["id"] in selected])
    output["pipeline_fijo_questions"] = fixed_rows
    output["cost"]["pipeline_fijo"] = fixed_cost
    checkpoint()
    if args.run_behavior_cases:
        output["behavior_cases"] = _run_cases()
        checkpoint()
    for label, dataset in (("pipeline_fijo", [row for row in fixed_rows if not row.get("error")]),
                           ("agente", [row for row in rows if not row.get("error")])):
        try:
            output["ragas"][label] = _ragas(dataset)
        except Exception as error:
            output["ragas"][label] = {"error": f"{type(error).__name__}: {str(error)[:500]}"}
            print(f"Ragas {label}: ERROR {type(error).__name__}: {str(error)[:180]}")
        checkpoint()
    print(f"Resultados guardados en {path}")


def _averages(rows):
    if not rows: return {}
    return {key: round(sum(r[key] for r in rows) / len(rows), 3)
            for key in ("llm_calls", "tokens", "elapsed_seconds")}


def _fixed_pipeline_cost(rows):
    """Repite las mismas preguntas en el RAG fijo e instrumenta llamadas/tokens de Groq."""
    original = rag._invoke_with_retry
    measured = {"llm_calls": 0, "tokens": 0}
    def tracked(llm, messages, *args, **kwargs):
        response = original(llm, messages, *args, **kwargs)
        measured["llm_calls"] += 1
        usage = getattr(response, "usage_metadata", None) or {}
        measured["tokens"] += int(usage.get("total_tokens", 0) or 0)
        return response
    timings, outputs = [], []
    rag._invoke_with_retry = tracked
    try:
        for row in rows:
            measured["llm_calls"] = measured["tokens"] = 0
            started = time.perf_counter()
            try:
                result = rag.answer_question(row["question"], history=[])
            except Exception as error:
                elapsed = time.perf_counter() - started
                outputs.append({"id": row["id"], "user_input": row["question"],
                                "error": f"{type(error).__name__}: {str(error)[:300]}",
                                "llm_calls": measured["llm_calls"], "tokens": measured["tokens"],
                                "elapsed_seconds": round(elapsed, 3)})
                continue
            elapsed = time.perf_counter() - started
            timings.append({"llm_calls": measured["llm_calls"], "tokens": measured["tokens"],
                            "elapsed_seconds": elapsed})
            outputs.append({"id": row["id"], "user_input": row["question"], "response": result["answer"],
                            "retrieved_contexts": result["contexts"], "reference": row["ground_truth"],
                            "llm_calls": measured["llm_calls"], "tokens": measured["tokens"],
                            "elapsed_seconds": round(elapsed, 3)})
    finally:
        rag._invoke_with_retry = original
    return outputs, _averages(timings)


def _ragas(rows):
    if not rows: return {}
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig
    from langchain_groq import ChatGroq
    from rag import _get_embeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    metrics = {"faithfulness": Faithfulness(), "answer_relevancy": ResponseRelevancy(strictness=1),
               "context_precision": LLMContextPrecisionWithReference(), "context_recall": LLMContextRecall()}
    judge = ChatGroq(model=os.getenv("RAGAS_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")),
                     temperature=0.0, max_tokens=int(os.getenv("RAGAS_MAX_TOKENS", "2048")),
                     api_key=os.getenv("GROQ_API_KEY"))
    llm = LangchainLLMWrapper(judge)
    embeddings = LangchainEmbeddingsWrapper(_get_embeddings(os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")))
    for metric in metrics.values():
        metric.llm = llm
        if hasattr(metric, "embeddings"): metric.embeddings = embeddings
        metric.init(RunConfig(timeout=240, max_retries=3,
                              max_workers=int(os.getenv("RAGAS_CONCURRENCY", "3"))))
    async def score():
        out = {}
        semaphore = asyncio.Semaphore(int(os.getenv("RAGAS_CONCURRENCY", "3")))
        for row in rows: out[row["id"]] = {}
        for name, metric in metrics.items():
            async def score_one(row):
                sample = SingleTurnSample(user_input=row["user_input"], response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"], reference=row["reference"])
                async with semaphore:
                    try:
                        out[row["id"]][name] = float(await metric._single_turn_ascore(sample, []))
                    except Exception as error:
                        out[row["id"]][name] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            await asyncio.gather(*(score_one(row) for row in rows))
        return out
    per_question = asyncio.run(score())
    averages = {}
    for name in metrics:
        values = [row[name] for row in per_question.values() if isinstance(row.get(name), (int, float))]
        if values: averages[name] = round(sum(values) / len(values), 4)
    return {"averages": averages, "per_question": per_question}


def _run_cases():
    cases = json.loads((ROOT / "evaluation" / "behavior_cases.json").read_text(encoding="utf-8"))
    out = []
    for case in cases:
        try:
            result = run_agent(case["question"])
        except Exception as error:
            out.append({**case, "error": f"{type(error).__name__}: {str(error)[:300]}", "passed": False})
            print(f"Comportamiento {case['id']}: ERROR {type(error).__name__}: {str(error)[:160]}")
            continue
        used = [call["name"] for call in result["tool_calls"]]
        passed = all(name in used for name in case["required_tools"])
        if case["id"] == 1: passed = not used and result["llm_calls"] == 0
        if case["id"] == 6: passed = "buscar_en_base" in used and "buscar_web" not in used
        if case["id"] == 7:
            text = result["answer"].casefold()
            passed = ("no encontré información sobre esto en la base de conocimientos." in text
                      or "no puedo entregar" in text or "no puedo mostrar" in text)
        out.append({**case, "obtained": result["answer"], "tools_used": result["tool_calls"],
                    "llm_calls": result["llm_calls"], "passed": bool(passed)})
    return out


if __name__ == "__main__": main()
