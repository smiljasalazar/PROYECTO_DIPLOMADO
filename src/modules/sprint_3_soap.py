"""
Sprint 3: Generador de Notas SOAP
Convierte notas clínicas libres en formato estructurado SOAP.
Usa Ollama (Llama2) para generación rápida.
"""

import streamlit as st
import requests


def generate_with_ollama(prompt: str) -> str:
    """Generate text using Ollama (Llama2)."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400
                }
            },
            timeout=180
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error Ollama: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Ollama no está corriendo. Ejecuta: ollama serve"
    except requests.exceptions.ReadTimeout:
        return "⏳ Timeout: Llama2 tardó demasiado."
    except Exception as e:
        return f"Error: {str(e)}"


# Example clinical notes
EXAMPLES = {
    "Dolor torácico": """Paciente masculino de 65 años acude por dolor en el pecho de 3 días de evolución. 
Dolor opresivo que empeora con el esfuerzo. Antecedentes: HTA, DM2. Toma losartán 50mg y metformina 850mg.
Examen: PA 150/95, FC 88, SpO2 96%. Auscultación cardíaca normal. ECG con cambios inespecíficos de ST.""",
    
    "Dolor abdominal": """Paciente femenina de 45 años con dolor abdominal de 2 días.
Localizado en hipocondrio derecho, tipo cólico. Náuseas y un episodio de vómito.
Sin fiebre. Antecedente de litiasis biliar. Examen: Murphy positivo, sin irritación peritoneal.
SV: PA 120/80, FC 78, T 36.8°C.""",
    
    "Infección respiratoria": """Paciente de 28 años con tos productiva de 5 días.
Expectoración amarillenta, fiebre 38.5°C, odinofagia, malestar general. No disnea.
Examen: Faringe eritematosa, amígdalas hipertróficas. Pulmones claros.
SV: PA 110/70, FC 82, SpO2 98%, T 37.8°C."""
}


def create_soap_prompt(clinical_note: str) -> str:
    """Create the SOAP generation prompt."""
    return f"""Eres un asistente médico experto. Convierte la siguiente nota clínica en formato SOAP estructurado.

NOTA CLÍNICA:
{clinical_note}

FORMATO SOAP:
S (Subjetivo): [Síntomas y quejas del paciente]
O (Objetivo): [Hallazgos del examen físico y signos vitales]
A (Evaluación): [Diagnóstico o impresión diagnóstica]
P (Plan): [Tratamiento y seguimiento]

NOTA SOAP:"""


def parse_soap(text: str) -> dict:
    """Parse generated text into SOAP sections."""
    sections = {
        "S": "",
        "O": "",
        "A": "",
        "P": ""
    }
    
    # Try to extract each section
    import re
    
    patterns = {
        "S": r"S\s*[\(:].*?[:\)]\s*(.+?)(?=O\s*[\(:]|$)",
        "O": r"O\s*[\(:].*?[:\)]\s*(.+?)(?=A\s*[\(:]|$)",
        "A": r"A\s*[\(:].*?[:\)]\s*(.+?)(?=P\s*[\(:]|$)",
        "P": r"P\s*[\(:].*?[:\)]\s*(.+?)$"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
    
    return sections


def audit_soap(sections: dict) -> dict:
    """Audit SOAP note completeness."""
    results = {}
    
    labels = {
        "S": "Subjetivo",
        "O": "Objetivo", 
        "A": "Evaluación",
        "P": "Plan"
    }
    
    for key, label in labels.items():
        has_content = len(sections.get(key, "")) > 10
        results[label] = {
            "present": has_content,
            "length": len(sections.get(key, ""))
        }
    
    results["complete"] = all(r["present"] for r in results.values() if isinstance(r, dict))
    results["score"] = sum(1 for r in results.values() if isinstance(r, dict) and r["present"]) / 4
    
    return results


def render_input_section():
    """Render input section."""
    
    st.markdown("### 📝 Nota Clínica de Entrada")
    
    tab1, tab2 = st.tabs(["✏️ Escribir", "📋 Ejemplos"])
    
    with tab1:
        clinical_note = st.text_area(
            "Ingresa la nota clínica",
            placeholder="Paciente de X años con...",
            height=200,
            key="custom_note"
        )
    
    with tab2:
        example_key = st.selectbox(
            "Selecciona un ejemplo",
            options=list(EXAMPLES.keys())
        )
        clinical_note = EXAMPLES[example_key]
        st.text_area(
            "Nota seleccionada",
            value=clinical_note,
            height=200,
            disabled=True,
            key="example_note"
        )
    
    return clinical_note


def render_soap_output(sections: dict, audit: dict):
    """Render SOAP output with audit."""
    
    st.markdown("### 📋 Nota SOAP Generada")
    
    # Audit summary
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if audit["complete"]:
            st.success("✅ Nota completa")
        else:
            st.warning("⚠️ Nota incompleta")
    with col2:
        st.metric("Completitud", f"{audit['score']:.0%}")
    
    st.divider()
    
    # SOAP sections
    section_config = {
        "S": {"icon": "🗣️", "title": "Subjetivo", "color": "blue"},
        "O": {"icon": "🔬", "title": "Objetivo", "color": "green"},
        "A": {"icon": "🩺", "title": "Evaluación", "color": "orange"},
        "P": {"icon": "📋", "title": "Plan", "color": "violet"}
    }
    
    for key, config in section_config.items():
        content = sections.get(key, "")
        
        with st.container():
            st.markdown(f"**{config['icon']} {config['title']}**")
            
            if content:
                st.info(content)
            else:
                st.warning("Sección no detectada")
        
        st.divider()
    
    # Audit details
    with st.expander("🔍 Auditoría detallada"):
        for section, data in audit.items():
            if isinstance(data, dict):
                status = "✅" if data["present"] else "❌"
                st.write(f"{status} **{section}**: {data['length']} caracteres")


def render(translations: dict, lang: str):
    """Main render function for Sprint 3."""
    
    # Description
    st.markdown("""
    Convierte notas clínicas en texto libre al formato estándar **SOAP**
    usando un modelo de generación de lenguaje (LLM).
    """)
    
    # SOAP explanation
    with st.expander("ℹ️ ¿Qué es el formato SOAP?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **S**ubjective: Lo que reporta el paciente
            - **O**bjective: Hallazgos del examen físico
            """)
        with col2:
            st.markdown("""
            - **A**ssessment: Evaluación/diagnóstico
            - **P**lan: Plan de tratamiento
            """)
    
    st.divider()
    
    # Input
    clinical_note = render_input_section()
    
    # Generate button
    if st.button("🔄 Generar Nota SOAP", type="primary", use_container_width=True):
        if clinical_note.strip():
            with st.spinner("⏳ Generando con Llama2 (Ollama)..."):
                prompt = create_soap_prompt(clinical_note)
                
                # Generate with Ollama
                soap_text = generate_with_ollama(prompt)
                
                # Check for errors
                if soap_text.startswith("❌") or soap_text.startswith("⏳") or soap_text.startswith("Error"):
                    st.error(soap_text)
                else:
                    # Parse and audit
                    sections = parse_soap(soap_text)
                    audit = audit_soap(sections)
                    
                    # Store in session
                    st.session_state.soap_result = {
                        "sections": sections,
                        "audit": audit,
                        "raw": soap_text
                    }
        else:
            st.warning("⚠️ Ingresa una nota clínica")
    
    # Display results
    if "soap_result" in st.session_state:
        st.divider()
        render_soap_output(
            st.session_state.soap_result["sections"],
            st.session_state.soap_result["audit"]
        )
        
        # Export
        with st.expander("📥 Exportar"):
            st.download_button(
                "⬇️ Descargar como texto",
                data=st.session_state.soap_result["raw"],
                file_name="nota_soap.txt",
                mime="text/plain"
            )
    
    # Ethics
    st.divider()
    with st.expander("⚠️ Consideraciones Éticas"):
        st.markdown("""
        - 🏥 Las notas generadas **deben ser revisadas** por el médico
        - 📝 No reemplazan la documentación clínica formal
        - 🔒 No ingresar datos reales de pacientes sin anonimización
        - 🧪 Sistema experimental para fines educativos
        """)
    
    st.caption("📓 Notebook: `notebooks/05_soap_generator.ipynb`")
