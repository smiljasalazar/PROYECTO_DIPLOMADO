"""
Test script para verificar Qwen 0.5B en tu Mac M1
"""
import time
import torch
from transformers import pipeline

print("=" * 50)
print("🧪 TEST: Qwen 0.5B en tu máquina")
print("=" * 50)

# Detectar dispositivo
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"\n🔧 Dispositivo detectado: {device}")
print(f"📱 MPS disponible: {torch.backends.mps.is_available()}")
print(f"🖥️  CUDA disponible: {torch.cuda.is_available()}")

# Cargar modelo
print("\n⏳ Cargando Qwen 0.5B...")
start_load = time.time()

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    device=device if device != "cpu" else -1,
    trust_remote_code=True,
    model_kwargs={"low_cpu_mem_usage": True}
)

load_time = time.time() - start_load
print(f"✅ Modelo cargado en {load_time:.1f} segundos")

# Test: Generar respuesta
print("\n⏳ Generando respuesta para 'Hola'...")
start_gen = time.time()

result = generator(
    "Hola, ¿cómo estás?",
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    pad_token_id=generator.tokenizer.eos_token_id
)

gen_time = time.time() - start_gen

print(f"\n{'=' * 50}")
print("📝 RESPUESTA:")
print(result[0]["generated_text"])
print(f"{'=' * 50}")
print(f"\n⏱️  Tiempo de generación: {gen_time:.1f} segundos")
print(f"✅ Todo funcionó correctamente!")
