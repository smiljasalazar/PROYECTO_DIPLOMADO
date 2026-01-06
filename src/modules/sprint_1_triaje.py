"""
Sprint 1: Triaje Zero-Shot
Clasificación de mensajes de pacientes en categorías de urgencia.
"""

import streamlit as st

# Lazy load heavy modules
_classifier = None

def get_classifier():
    """Lazy load the classification pipeline."""
    global _classifier
    if _classifier is None:
        with st.spinner("⏳ Cargando modelo de triaje..."):
            from transformers import pipeline
            _classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            )
    return _classifier

# Default labels
DEFAULT_LABELS = [
    "Urgencia Vital (Roja)",
    "Urgencia Moderada (Amarilla)", 
    "Consulta Médica No Urgente",
    "Consulta Administrativa",
    "Renovación de Receta"
]

# Example messages
EXAMPLES = [
    "Tengo un dolor muy fuerte en el pecho y se me duerme el brazo izquierdo.",
    "Hola, necesito saber si el doctor Martínez atiende los jueves.",
    "Se me acabó la pastilla de la presión, necesito la receta.",
    "Mi hijo tiene fiebre de 38 desde ayer y llora mucho.",
    "Siento presión muy fuerte en el pecho y me falta el aire.",
]


def render_config_panel():
    """Render configuration panel."""
    with st.expander("⚙️ Configuración", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            threshold = st.slider(
                "Umbral de confianza",
                min_value=0.3,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="Clasificaciones por debajo se marcarán como baja confianza"
            )
        
        with col2:
            use_custom = st.checkbox("Usar etiquetas personalizadas")
            
            if use_custom:
                custom_labels = st.text_area(
                    "Etiquetas (una por línea)",
                    value="\n".join(DEFAULT_LABELS),
                    height=100
                )
                labels = [l.strip() for l in custom_labels.split("\n") if l.strip()]
            else:
                labels = DEFAULT_LABELS
        
        return labels, threshold


def render_input_panel(labels: list):
    """Render input panel with tabs."""
    
    tab1, tab2, tab3 = st.tabs(["✏️ Escribir", "📋 Ejemplos", "📄 Múltiples"])
    
    messages_to_classify = []
    
    with tab1:
        single_msg = st.text_area(
            "Mensaje del paciente",
            placeholder="Ej: Tengo un dolor fuerte en el pecho...",
            height=100,
            key="single_input"
        )
        if st.button("🔍 Clasificar", key="btn_single", type="primary"):
            if single_msg.strip():
                messages_to_classify = [single_msg.strip()]
            else:
                st.warning("⚠️ Escribe un mensaje")
    
    with tab2:
        st.markdown("**Selecciona ejemplos:**")
        selected = []
        for i, example in enumerate(EXAMPLES):
            if st.checkbox(example[:50] + "...", key=f"ex_{i}"):
                selected.append(example)
        
        if st.button("🔍 Clasificar seleccionados", key="btn_examples"):
            if selected:
                messages_to_classify = selected
            else:
                st.warning("⚠️ Selecciona al menos un ejemplo")
    
    with tab3:
        multi_msg = st.text_area(
            "Múltiples mensajes (uno por línea)",
            placeholder="Mensaje 1\nMensaje 2\nMensaje 3",
            height=150,
            key="multi_input"
        )
        if st.button("🔍 Clasificar todos", key="btn_multi"):
            lines = [l.strip() for l in multi_msg.split("\n") if l.strip()]
            if lines:
                messages_to_classify = lines
            else:
                st.warning("⚠️ Ingresa al menos un mensaje")
    
    return messages_to_classify


def classify_messages(messages: list, labels: list, threshold: float):
    """Classify messages and display results."""
    
    classifier = get_classifier()
    
    st.markdown("### 📊 Resultados")
    
    progress = st.progress(0)
    
    results = []
    
    for i, msg in enumerate(messages):
        progress.progress((i + 1) / len(messages))
        
        # Classify
        result = classifier(msg, labels)
        
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        results.append({
            "mensaje": msg,
            "categoria": top_label,
            "confianza": top_score,
            "all_scores": dict(zip(result["labels"], result["scores"]))
        })
    
    progress.empty()
    
    # Display results
    for r in results:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**📩 {r['mensaje'][:60]}...**" if len(r['mensaje']) > 60 else f"**📩 {r['mensaje']}**")
            
            with col2:
                # Color based on urgency
                if "Vital" in r["categoria"] or "Roja" in r["categoria"]:
                    st.error(f"🚨 {r['categoria']}")
                elif "Moderada" in r["categoria"] or "Amarilla" in r["categoria"]:
                    st.warning(f"⚠️ {r['categoria']}")
                else:
                    st.info(f"ℹ️ {r['categoria']}")
            
            with col3:
                # Confidence indicator
                if r["confianza"] >= threshold:
                    st.success(f"{r['confianza']:.0%}")
                else:
                    st.warning(f"{r['confianza']:.0%} ⚠️")
            
            # Expandable details
            with st.expander("Ver detalles"):
                for label, score in r["all_scores"].items():
                    st.progress(score, text=f"{label}: {score:.1%}")
            
            st.divider()
    
    return results


def render_ethics_panel():
    """Render ethical considerations."""
    with st.expander("⚠️ Consideraciones Éticas"):
        st.markdown("""
        - 🏥 Este sistema es **experimental** y NO reemplaza el juicio clínico
        - 👨‍⚕️ Todas las clasificaciones deben ser **validadas por profesionales**
        - 🔒 No ingresar datos reales de pacientes sin anonimización
        - 📊 El modelo puede tener sesgos en ciertos tipos de mensajes
        """)


def render(translations: dict, lang: str):
    """Main render function for Sprint 1."""
    
    # Description
    st.markdown("""
    Clasifica mensajes de pacientes en categorías de urgencia usando
    un modelo **Zero-Shot** (sin entrenamiento previo en datos de triaje).
    """)
    
    st.divider()
    
    # Configuration
    labels, threshold = render_config_panel()
    
    # Input
    messages = render_input_panel(labels)
    
    # Classification
    if messages:
        classify_messages(messages, labels, threshold)
    
    # Ethics
    st.divider()
    render_ethics_panel()
    
    # Notebook link
    st.caption("📓 Notebook: `notebooks/01_triaje_zeroshot.ipynb`")
