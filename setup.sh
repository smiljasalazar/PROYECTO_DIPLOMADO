#!/bin/bash
# ============================================
# 🔧 Script de Instalación Automática
# Asistente Clínico Inteligente - UPCH
# ============================================

echo "════════════════════════════════════════════"
echo "🏥 Instalador del Asistente Clínico UPCH"
echo "════════════════════════════════════════════"
echo ""

# Check Python version
echo "📌 Verificando Python..."
python3 --version || { echo "❌ Python 3 no encontrado. Instálalo primero."; exit 1; }

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activando entorno virtual..."
source .venv/bin/activate

# Install dependencies
echo "📥 Instalando dependencias (puede tardar 2-5 min)..."
pip install --upgrade pip
pip install -r requirements.txt

# Install spaCy models
echo ""
echo "🧠 Descargando modelos de NLP (puede tardar 5-10 min)..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

echo ""
echo "════════════════════════════════════════════"
echo "✅ ¡Instalación completada!"
echo "════════════════════════════════════════════"
echo ""
echo "Para ejecutar la aplicación:"
echo ""
echo "  1. Activa el entorno: source .venv/bin/activate"
echo "  2. Ejecuta: streamlit run src/streamlit_app.py"
echo "  3. Abre en navegador: http://localhost:8501"
echo ""
echo "════════════════════════════════════════════"
