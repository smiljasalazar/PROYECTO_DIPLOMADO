# 🏥 ACIE - Asistente Clínico Inteligente con Embeddings

Sistema didáctico de NLP Médico desarrollado para el curso **Transformers en Salud** de la UPCH.

## 📚 Módulos

| Sprint | Nombre | Descripción |
|--------|--------|-------------|
| 1 | 🎯 Triaje Zero-Shot | Clasificación de urgencia sin entrenamiento |
| 2 | 🔖 NER Médico | Extracción de entidades clínicas |
| 3 | 📝 Generador SOAP | Notas clínicas estructuradas |
| 4 | 💬 RAG Clínico | Preguntas sobre documentos médicos |

---

## 🚀 Instalación Rápida

### Requisitos
- Python 3.10+
- [Ollama](https://ollama.ai) (para Sprint 4)
- 8GB RAM mínimo

### Paso 1: Clonar repositorio
```bash
git clone https://github.com/BryPhysic/Proyecto_T_L.git
cd Proyecto_T_L
```

### Paso 2: Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# o en Windows: .venv\Scripts\activate
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Instalar Ollama + Llama2 (para Sprint 4)
```bash
# macOS
brew install ollama

# Descargar modelo
ollama pull llama2
```

### Paso 5: Ejecutar
```bash
streamlit run src/streamlit_app.py
```

Abre http://localhost:8501 en tu navegador.

---

## 📦 Modos de Uso

### 🚀 Modo LITE (Recomendado para empezar)
- ✅ **No requiere descargas adicionales**
- ✅ Sube tus propios PDFs/TXTs
- ✅ Funciona con Ollama local
- Sprint 4: Solo busca en tus documentos

### 📚 Modo COMPLETO (Con base de datos UMLS)
1. Descarga `ACIE_datos_completos.zip` (~12GB) desde:
   - [Link de Google Drive - pendiente]
   
2. Descomprime en la carpeta del proyecto:
```bash
unzip ACIE_datos_completos.zip -d Datasets/
```

3. En Sprint 4, selecciona "📚 Completo (UMLS)" para buscar también en la base de conocimiento médico.

---

## 📁 Estructura del Proyecto

```
Proyecto_T_L/
├── src/
│   ├── streamlit_app.py      # App principal
│   ├── modules/              # Páginas de cada Sprint
│   └── utils/                # Procesadores (NER, RAG, etc.)
├── notebooks/                # Notebooks didácticos
├── data/examples/            # Datos de ejemplo
├── Datasets/                 # Bases de datos (no en GitHub)
│   ├── chromadb_umls/        # Base UMLS (modo completo)
│   └── rag_documents/        # Tus documentos
└── requirements.txt          # Dependencias
```

---

## 🔧 Solución de Problemas

### "Ollama no está corriendo"
```bash
ollama serve  # Inicia el servidor
```

### "No encontré información relevante"
- Sube un documento PDF/TXT primero
- Haz preguntas relacionadas al contenido del documento

### Sprint 4 muy lento
- La primera respuesta tarda 1-2 min (carga del modelo)
- Las siguientes son más rápidas
- Usa "🔍 Solo Búsqueda" si no quieres esperar

---

## 👥 Créditos

- **Curso**: Transformers en Salud - UPCH
- **Versión**: 2.0 - Enero 2026
- **Autor**: BryPhysic

---

## 📄 Licencia

Uso educativo - UPCH
