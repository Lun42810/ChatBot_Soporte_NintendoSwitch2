from rag import retrieve, RAGConfig

cfg = RAGConfig.from_env()
hits = retrieve("precauciones cámara", cfg)

print(f"Total de resultados: {len(hits)}\n")

for doc, meta in hits[:3]:
    # Si meta es un número (score), imprime el documento y el score
    if isinstance(meta, (float, int)):
        print(f"Score: {meta}")
        print(doc.page_content[:300])
    # Si meta es un diccionario, imprime los metadatos
    elif isinstance(meta, dict):
        print(f"Fuente: {meta.get('source', 'Desconocido')} (pág. {meta.get('page', '?')})")
        print(doc.page_content[:300])
    print("-" * 80)
