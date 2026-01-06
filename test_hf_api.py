"""
Test HuggingFace Inference API
"""
import requests
import os

# Tu token de HuggingFace
HF_TOKEN = input("Ingresa tu token HuggingFace (hf_...): ").strip()

if not HF_TOKEN:
    print("❌ Token requerido")
    exit(1)

print("\n" + "="*50)
print("🧪 TEST: HuggingFace Inference API")
print("="*50)

# Probar diferentes URLs y modelos
tests = [
    # Nuevo endpoint router
    {
        "name": "Phi-3 Mini (router)",
        "url": "https://router.huggingface.co/hf-inference/models/microsoft/Phi-3-mini-4k-instruct",
    },
    # Endpoint antiguo (de respaldo)
    {
        "name": "Phi-3 Mini (api-inference)",
        "url": "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
    },
    # Modelo simple
    {
        "name": "GPT-2 (test básico)",
        "url": "https://api-inference.huggingface.co/models/gpt2",
    },
]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

for test in tests:
    print(f"\n📡 Probando: {test['name']}")
    print(f"   URL: {test['url'][:60]}...")
    
    try:
        response = requests.post(
            test["url"],
            headers=headers,
            json={"inputs": "Hola, ¿cómo estás?"},
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Respuesta: {str(result)[:200]}...")
        elif response.status_code == 401:
            print("   ❌ Token inválido")
        elif response.status_code == 404:
            print("   ❌ Modelo no encontrado en API")
        elif response.status_code == 503:
            print("   ⏳ Modelo cargando, intenta de nuevo")
        else:
            print(f"   ❌ Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*50)
