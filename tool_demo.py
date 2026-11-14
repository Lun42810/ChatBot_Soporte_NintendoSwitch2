"""Demuestra la búsqueda de la base sin inicializar ni llamar al LLM."""
import argparse
import json

from agent import invoke_base_tool_without_llm

parser = argparse.ArgumentParser()
parser.add_argument("consulta", nargs="?", default="¿Cómo se conecta la cámara a Nintendo Switch 2?")
args = parser.parse_args()
print(json.dumps(invoke_base_tool_without_llm(args.consulta), ensure_ascii=False, indent=2))
