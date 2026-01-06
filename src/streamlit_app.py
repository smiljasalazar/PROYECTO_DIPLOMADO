"""
ACIE - Asistente Clínico Inteligente y Explicable
=================================================
Professional Streamlit UI with native elements only.
Optimized for performance with lazy loading.
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="ACIE - Asistente Clínico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SPRINTS = [
    {"num": 1, "icon": "🎯", "name": "Triaje Zero-Shot", "status": "ready", "color": "green"},
    {"num": 2, "icon": "🔖", "name": "NER Médico", "status": "ready", "color": "green"},
    {"num": 3, "icon": "📝", "name": "Generador SOAP", "status": "ready", "color": "green"},
    {"num": 4, "icon": "💬", "name": "RAG Clínico", "status": "ready", "color": "green"},
]

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session():
    """Initialize session state."""
    if "sprint" not in st.session_state:
        st.session_state.sprint = 1
    if "lang" not in st.session_state:
        st.session_state.lang = "es"

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render professional sidebar."""
    with st.sidebar:
        # Logo/Header
        st.markdown("## 🏥 ACIE")
        st.caption("Asistente Clínico Inteligente")
        
        st.divider()
        
        # Language toggle
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇪🇸 ES", use_container_width=True, 
                        type="primary" if st.session_state.lang == "es" else "secondary"):
                st.session_state.lang = "es"
                st.rerun()
        with col2:
            if st.button("🇬🇧 EN", use_container_width=True,
                        type="primary" if st.session_state.lang == "en" else "secondary"):
                st.session_state.lang = "en"
                st.rerun()
        
        st.divider()
        
        # Sprint Navigation
        st.markdown("### 📋 Módulos")
        
        for sprint in SPRINTS:
            # Status badge
            if sprint["status"] == "ready":
                badge = "✅"
            elif sprint["status"] == "beta":
                badge = "🧪"
            else:
                badge = "⏳"
            
            # Sprint button
            is_selected = st.session_state.sprint == sprint["num"]
            label = f"{sprint['icon']} {sprint['name']}"
            
            if st.button(
                label,
                key=f"nav_{sprint['num']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.sprint = sprint["num"]
                st.rerun()
            
            st.caption(f"   {badge} Sprint {sprint['num']}")
        
        st.divider()
        
        # Progress
        st.markdown("### 📊 Progreso")
        completed = sum(1 for s in SPRINTS if s["status"] == "ready")
        st.progress(completed / len(SPRINTS))
        st.caption(f"{completed}/{len(SPRINTS)} módulos listos")
        
        # Footer
        st.divider()
        st.caption("🎓 UPCH - Transformers en Salud")
        st.caption("v2.0 - Enero 2026")

# ============================================================================
# MAIN CONTENT
# ============================================================================

def render_header():
    """Render main header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        current = SPRINTS[st.session_state.sprint - 1]
        st.title(f"{current['icon']} {current['name']}")
        
        # Status indicator
        if current["status"] == "ready":
            st.success("✅ Módulo disponible")
        elif current["status"] == "beta":
            st.warning("🧪 Versión beta")
        else:
            st.info("⏳ En desarrollo")
    
    with col2:
        st.metric("Sprint", st.session_state.sprint, delta=None)

def render_sprint():
    """Render current sprint content with lazy loading."""
    sprint_num = st.session_state.sprint
    lang = st.session_state.lang
    
    # Simple translations
    translations = {
        "es": {
            "coming_soon": "Próximamente",
            "see_notebook": "Ver notebook",
            "results": "Resultados",
            "abbreviations": "Abreviaturas",
        },
        "en": {
            "coming_soon": "Coming soon",
            "see_notebook": "See notebook",
            "results": "Results",
            "abbreviations": "Abbreviations",
        }
    }
    
    t = translations[lang]
    
    # Lazy load sprint pages
    if sprint_num == 1:
        from modules import sprint_1_triaje
        sprint_1_triaje.render(t, lang)
    elif sprint_num == 2:
        from modules import sprint_2_ner
        sprint_2_ner.render(t, lang)
    elif sprint_num == 3:
        from modules import sprint_3_soap
        sprint_3_soap.render(t, lang)
    elif sprint_num == 4:
        from modules import sprint_4_rag
        sprint_4_rag.render(t, lang)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application."""
    init_session()
    render_sidebar()
    render_header()
    st.divider()
    render_sprint()

if __name__ == "__main__":
    main()
