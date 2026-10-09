"""Construye el índice vectorial. Uso: python ingest.py [--rebuild]"""
import argparse

from dotenv import load_dotenv

load_dotenv()
from rag import RAGConfig, build_store  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="borra y recrea el índice de la configuración actual")
    args = parser.parse_args()
    cfg = RAGConfig.from_env()
    store = build_store(cfg, rebuild=args.rebuild)
    print(f"Colección '{cfg.collection_name}': {store._collection.count()} fragmentos indexados.")
