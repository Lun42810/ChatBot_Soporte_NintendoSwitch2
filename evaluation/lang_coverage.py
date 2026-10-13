"""Cuenta, por manual, cuántos fragmentos hay en cada idioma (según el detector de rag.py).
No llama a ninguna API ni gasta tokens.

Uso (desde la raíz del proyecto):  python evaluation/lang_coverage.py
Sirve para comprobar que el filtro languages="es" no deja sin texto a ningún manual.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from rag import RAGConfig, load_documents, split_documents  # noqa: E402

cfg = RAGConfig(languages="", header=False)     # sin filtro: queremos ver todo
chunks = split_documents(load_documents(), cfg)

by_source = defaultdict(Counter)
examples = defaultdict(list)
for c in chunks:
    src = Path(c.metadata.get("source", "?")).name.replace("NSwitch2_", "").replace("_EUR.pdf", "")
    lang = c.metadata.get("lang", "?")
    by_source[src][lang] += 1
    if lang != "es" and len(examples[src]) < 40:
        examples[src].append((lang, c.page_content.strip().replace("\n", " ")))

print(f"{'manual':48s} {'total':>5s} {'es':>4s} {'unknown':>8s} {'otros':>6s}")
for src, cnt in sorted(by_source.items()):
    total = sum(cnt.values())
    es, unk = cnt.get("es", 0), cnt.get("unknown", 0)
    flag = "   <-- pocos en español" if es < 3 else ""
    print(f"{src:48s} {total:5d} {es:4d} {unk:8d} {total - es - unk:6d}{flag}")

print("\nFragmentos 'unknown' (no clasificados; el filtro los descarta) de los manuales con pocos en español:")
for src, cnt in sorted(by_source.items()):
    if cnt.get("es", 0) < 3:
        print(f"\n== {src}")
        shown = 0
        for lang, text in examples[src]:
            if lang == "unknown" and shown < 4:
                print(f"  - {text[:200]}")
                shown += 1
        if shown == 0:
            print("  (no hay fragmentos 'unknown'; el texto en español no está en el PDF o se clasificó como otro idioma)")
