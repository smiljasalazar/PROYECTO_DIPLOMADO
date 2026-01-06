# 🚀 Guía de Instalación

## Requisitos Mínimos

| Requisito | Mínimo | Recomendado |
|-----------|--------|-------------|
| **RAM** | 4 GB | 8 GB |
| **Disco** | 2 GB | 5 GB |
| **Python** | 3.10 | 3.10-3.11 |
| **OS** | macOS/Linux/Windows | - |

---

## Instalación Rápida (Mac/Linux)

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/Proyecto_T_L.git
cd Proyecto_T_L

# 2. Ejecutar instalador automático
chmod +x setup.sh
./setup.sh

# 3. Activar entorno y ejecutar
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

---

## Instalación Manual (Windows)

### Paso 1: Clonar repositorio
```bash
git clone https://github.com/TU_USUARIO/Proyecto_T_L.git
cd Proyecto_T_L
```

### Paso 2: Crear entorno virtual
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Instalar modelos NLP
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### Paso 5: Ejecutar
```bash
streamlit run src/streamlit_app.py
```

---

## Solución de Problemas

### "No module named X"
```bash
pip install -r requirements.txt
```

### "Model not found"
Ejecutar nuevamente los comandos del Paso 4.

### "Out of memory" en NER Avanzado
Usar solo el modo **"Básico"** en Sprint 2.

---

## Modos de Uso

| Modo | RAM | Descripción |
|------|-----|-------------|
| **Básico** | ~1 GB | NER sin UMLS (rápido) |
| **ChromaDB** | ~2 GB | NER + UMLS optimizado |
| **Legacy** | ~15 GB | NER + UMLS completo |

**Recomendación**: Usar modo "Básico" o "ChromaDB" para laptops normales.

---

## Contacto

Si tienes problemas, contacta al profesor o abre un Issue en GitHub.
