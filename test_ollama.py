"""
Test Ollama + ChromaDB RAG
"""
import requests

print("="*50)
print("🧪 TEST: Ollama + ChromaDB RAG")
print("="*50)

# 1. Test Ollama connection
print("\n1️⃣ Probando conexión a Ollama...")
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    if r.status_code == 200:
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"   ✅ Ollama conectado")
        print(f"   📦 Modelos disponibles: {models}")
    else:
        print(f"   ❌ Error: {r.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ Ollama no está corriendo!")
    print("   👉 Ejecuta: ollama serve")
    exit(1)

# 2. Test generation
print("\n2️⃣ Probando generación...")
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama2",  # Cambia a tu modelo
        "prompt": "Responde en español: ¿Qué es la hipertensión?",
        "stream": False,
        "options": {"num_predict": 100}
    },
    timeout=60
)

if response.status_code == 200:
    answer = response.json().get("response", "")
    print(f"   ✅ Respuesta recibida ({len(answer)} chars)")
    print(f"\n📝 Respuesta:\n{answer[:300]}...")
else:
    print(f"   ❌ Error: {response.status_code}")

print("\n" + "="*50)
print("✅ Ollama funcionando correctamente!")
