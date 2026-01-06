"""
Sprint 2: NER y Estructuración
Advanced Named Entity Recognition with multi-model approach and EntityLinker with UMLS.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

# Import NER processor
try:
    from utils.ner_processor import NERProcessor, export_to_json
except ImportError:
    st.error("⚠️ Error importing NER processor. Please check installation.")


# Clinical examples
EJEMPLOS_CLINICOS = {
    'es': [
        {
            'nombre': 'Caso 1: Paciente con antecedentes cardiovasculares',
            'texto': """Paciente varón de 65 años, acude a emergencia por cuadro de 3 días de evolución caracterizado por disnea de medianos esfuerzos y dolor torácico opresivo.
Antecedentes: Hipertensión arterial diagnosticada hace 10 años y Diabetes Mellitus tipo 2.
Actualmente en tratamiento con Losartán 50mg cada 12 horas y Metformina 850mg una vez al día.
Al examen físico: PA 150/90 mmHg, FC 95 lpm. Murmullo vesicular disminuido en bases.
Niega alergias a medicamentos conocidos. Se descarta infarto agudo de miocardio por enzimas cardiacas negativas."""
        },
        {
            'nombre': 'Caso 2: Consulta respiratoria',
            'texto': """Paciente de 45 años con tos seca de 2 semanas de evolución y fiebre intermitente.
Antecedente de asma bronquial desde la infancia. 
Tratamiento habitual con salbutamol inhalado. 
Niega contacto con pacientes COVID-19. Radiografía de tórax muestra infiltrado en base derecha."""
        }
    ],
    'en': [
        {
            'nombre': 'Case 1: Patient with cardiovascular history',
            'texto': """65-year-old male patient presents to emergency with 3-day history of dyspnea on moderate exertion and oppressive chest pain.
History: Arterial hypertension diagnosed 10 years ago and Type 2 Diabetes Mellitus.
Currently on treatment with Losartan 50mg every 12 hours and Metformin 850mg once daily.
Physical exam: BP 150/90 mmHg, HR 95 bpm. Decreased breath sounds at bases.
Denies known medication allergies. Acute myocardial infarction ruled out by negative cardiac enzymes."""
        },
        {
            'nombre': 'Case 2: Respiratory consultation',
            'texto': """45-year-old patient with dry cough for 2 weeks and intermittent fever.
History of bronchial asthma since childhood.
Current treatment with inhaled salbutamol.
Denies COVID-19 contact. Chest X-ray shows right base infiltrate."""
        }
    ]
}


@st.cache_resource
def load_basic_models():
    """Load basic NER models (cached)."""
    with st.spinner("⏳ Cargando modelos NER básicos..."):
        processor = NERProcessor(load_advanced=False)
    return processor


@st.cache_resource
def load_advanced_models(use_chromadb: bool = False):
    """
    Load advanced NER models with EntityLinker (cached).
    
    Args:
        use_chromadb: If True, use ChromaDB (Low RAM). Else use scispacy Linker (High RAM).
    """
    if use_chromadb:
        with st.spinner("⏳ Cargando NER con ChromaDB (Optimizado)..."):
            # Path to ChromaDB - adjusting to relative path for stability
            import os
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            chroma_path = os.path.join(base_path, "Datasets", "chromadb_umls")
            
            processor = NERProcessor(load_advanced=True, use_chromadb=True, chromadb_path=chroma_path)
    else:
        with st.spinner("⏳ Cargando EntityLinker FULL (Puede tardar y usar mucha RAM)..."):
            processor = NERProcessor(load_advanced=True, use_chromadb=False)
            
    return processor


def render_basic_results(resultados: Dict[str, Any], translations: dict):
    """Render results from basic multi-model NER."""
    
    st.markdown("---")
    st.subheader("📊 " + translations.get('results', 'Resultados'))
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Hugging Face",
            resultados['estadisticas']['total_hf'],
            help="Entidades detectadas por d4data/biomedical-ner-all"
        )
    
    with col2:
        st.metric(
            "SciBERT",
            resultados['estadisticas']['total_scibert'],
            help="Entidades detectadas por en_core_sci_scibert"
        )
    
    with col3:
        st.metric(
            "BC5CDR",
            resultados['estadisticas']['total_bc5cdr'],
            help="Enfermedades y químicos por en_ner_bc5cdr_md"
        )
    
    with col4:
        st.metric(
            translations.get('abbreviations', 'Abreviaturas'),
            len(resultados['abreviaturas']),
            help="Abreviaturas médicas detectadas"
        )
    
    # Comparative chart
    st.markdown("### 📈 Comparación de Modelos")
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Hugging Face', 'SciBERT', 'BC5CDR'],
            y=[
                resultados['estadisticas']['total_hf'],
                resultados['estadisticas']['total_scibert'],
                resultados['estadisticas']['total_bc5cdr']
            ],
            marker_color=['#3498db', '#2ecc71', '#e74c3c'],
            text=[
                resultados['estadisticas']['total_hf'],
                resultados['estadisticas']['total_scibert'],
                resultados['estadisticas']['total_bc5cdr']
            ],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Entidades Detectadas por Modelo",
        xaxis_title="Modelo NER",
        yaxis_title="Cantidad de Entidades",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results in tabs
    tab1, tab2, tab3 = st.tabs(["🤗 Hugging Face", "🧠 SciBERT", "🏥 BC5CDR"])
    
    with tab1:
        if resultados['entidades_huggingface']:
            df_hf = pd.DataFrame(resultados['entidades_huggingface'])
            st.dataframe(df_hf, use_container_width=True)
        else:
            st.info("No se detectaron entidades con este modelo")
    
    with tab2:
        if resultados['entidades_scibert']:
            for i, ent in enumerate(resultados['entidades_scibert'], 1):
                with st.expander(f"**{ent['texto']}** - {ent['tipo']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tipo:** {ent['tipo']}")
                        st.write(f"**Posición:** {ent['posicion']}")
                    with col2:
                        if ent['contexto'].get('temporalidad'):
                            st.write(f"⏰ **Temporalidad:** {ent['contexto']['temporalidad']}")
                        st.write(f"**Certeza:** {ent['contexto']['certeza']}")
                        if ent['contexto']['negacion']:
                            st.write("❌ **NEGADO**")
        else:
            st.info("No se detectaron entidades con este modelo")
    
    with tab3:
        if resultados['entidades_bc5cdr']:
            for i, ent in enumerate(resultados['entidades_bc5cdr'], 1):
                tipo_color = "🔴" if ent['tipo'] == "DISEASE" else "🔵"
                with st.expander(f"{tipo_color} **{ent['texto']}** - {ent['tipo']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tipo:** {ent['tipo']}")
                        st.write(f"**Posición:** {ent['posicion']}")
                    with col2:
                        if ent['contexto'].get('temporalidad'):
                            st.write(f"⏰ **Temporalidad:** {ent['contexto']['temporalidad']}")
                        st.write(f"**Certeza:** {ent['contexto']['certeza']}")
                        if ent['contexto']['negacion']:
                            st.write("❌ **NEGADO**")
        else:
            st.info("No se detectaron entidades con este modelo")


def render_advanced_results(resultados: Dict[str, Any], translations: dict):
    """Render results from advanced NER with EntityLinker."""
    
    st.markdown("---")
    st.subheader("📊 " + translations.get('results', 'Resultados'))
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Only show total concepts if available (Legacy mode)
        if 'total_conceptos_umls' in resultados['metadatos']:
            st.metric(
                "🗄️ Conceptos UMLS",
                f"{resultados['metadatos']['total_conceptos_umls']:,}",
                help="Total de conceptos médicos disponibles en UMLS"
            )
        else:
            st.metric(
                "🗄️ Modo",
                resultados['metadatos'].get('modo', 'ChromaDB'),
                help="Modo de procesamiento utilizado"
            )
    
    with col2:
        total_entities = sum(resultados['estadisticas'].values())
        st.metric(
            "🏷️ Entidades Totales",
            total_entities,
            help="Total de entidades detectadas"
        )
    
    with col3:
        st.metric(
            "🔤 Abreviaturas",
            len(resultados['abreviaturas']),
            help="Abreviaturas médicas detectadas"
        )
    
    # Category distribution chart
    if resultados['estadisticas']:
        st.markdown("### 📈 Distribución por Categoría")
        
        fig = px.pie(
            values=list(resultados['estadisticas'].values()),
            names=list(resultados['estadisticas'].keys()),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Entities by category
    st.markdown("### 🏥 Entidades por Categoría")
    
    categorias_iconos = {
        "ENFERMEDAD": "🦠",
        "SINTOMA": "🩺",
        "MEDICAMENTO": "💊",
        "ANATOMIA": "🫀",
        "PROCEDIMIENTO": "⚕️",
        "OTRO": "📋"
    }
    
    for categoria, entidades in resultados['entidades_por_categoria'].items():
        if entidades:
            icono = categorias_iconos.get(categoria, "📋")
            st.markdown(f"#### {icono} {categoria} ({len(entidades)})")
            
            for i, ent in enumerate(entidades, 1):
                # Check if entity has UMLS information
                if 'umls_id' in ent:
                    titulo = f"**{ent['texto_original']}** → {ent['nombre_normalizado']}"
                    with st.expander(titulo):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**🆔 UMLS CUI:** `{ent['umls_id']}`")
                            st.markdown(f"**📝 Nombre normalizado:** {ent['nombre_normalizado']}")
                            st.markdown(f"**🏷️ Tipos semánticos:** {', '.join(ent['tipos_semanticos'])}")
                            st.markdown(f"**✅ Confianza linking:** {ent['score_linking']:.3f}")
                            
                            # Definition
                            if ent.get('definicion') and ent['definicion'] != "Sin definición disponible":
                                st.markdown(f"**📖 Definición:**")
                                st.info(ent['definicion'])
                            
                            # Synonyms
                            if ent.get('sinonimos'):
                                st.markdown(f"**🔄 Sinónimos:** {', '.join(ent['sinonimos'][:3])}")
                        
                        with col2:
                            # Context
                            st.markdown("**Contexto:**")
                            if ent['contexto'].get('temporalidad'):
                                st.write(f"⏰ {ent['contexto']['temporalidad']}")
                            st.write(f"✓ {ent['contexto']['certeza']}")
                            if ent['contexto']['negacion']:
                                st.write("❌ NEGADO")
                        
                        # Alternatives
                        if ent.get('alternativas'):
                            st.markdown("**🔄 Conceptos alternativos:**")
                            for alt in ent['alternativas'][:2]:
                                st.write(f"• {alt['nombre']} (CUI: `{alt['umls_id']}`, score: {alt['score']:.3f})")
                
                else:
                    # Entity without UMLS linking
                    with st.expander(f"**{ent['texto_original']}** - {ent['tipo_ner']}"):
                        st.write(f"**Tipo NER:** {ent['tipo_ner']}")
                        st.write(f"**Posición:** {ent['posicion']}")
                        if ent['contexto'].get('temporalidad'):
                            st.write(f"⏰ **Temporalidad:** {ent['contexto']['temporalidad']}")
                        st.write(f"**Certeza:** {ent['contexto']['certeza']}")
                        if ent['contexto']['negacion']:
                            st.write("❌ **NEGADO**")


def render(translations: dict, lang: str):
    """
    Render the Sprint 2 NER page.
    
    Args:
        translations: Dictionary with UI translations
        lang: Current language code ('es' or 'en')
    """
    st.title("🔖 Sprint 2: NER y Estructuración" if lang == 'es' else "🔖 Sprint 2: NER and Structuring")
    
    # Introduction
    if lang == 'es':
        st.markdown("""
        ### Objetivo
        
        Extraer entidades clínicas de notas médicas y convertirlas en datos estructurados mediante 
        **Named Entity Recognition (NER)** avanzado con múltiples modelos y enriquecimiento semántico.
        
        Selecciona el modo de procesamiento:
        - **Modo Básico**: Comparación de 3 modelos NER especializados
        - **Modo Avanzado**: EntityLinker con UMLS (3.9M conceptos médicos)
        """)
    else:
        st.markdown("""
        ### Objective
        
        Extract clinical entities from medical notes and convert them into structured data using 
        advanced **Named Entity Recognition (NER)** with multiple models and semantic enrichment.
        
        Select processing mode:
        - **Basic Mode**: Comparison of 3 specialized NER models
        - **Advanced Mode**: EntityLinker with UMLS (3.9M medical concepts)
        """)
    
    st.markdown("---")
    
    # Mode selector
    modo_opciones = {
        'es': ["Básico", "Avanzado (ChromaDB)", "Avanzado (Legacy)"],
        'en': ["Basic", "Advanced (ChromaDB)", "Advanced (Legacy)"]
    }
    
    modo = st.radio(
        "🎯 " + ("Modo de Procesamiento" if lang == 'es' else "Processing Mode"),
        modo_opciones[lang],
        horizontal=True,
        help=("ChromaDB: Rápido y ligero | Legacy: Lento y pesado (UMLS en RAM)" if lang == 'es' 
              else "ChromaDB: Fast & Light | Legacy: Slow & Heavy (UMLS in RAM)")
    )
    
    is_advanced = "Avanzado" in modo or "Advanced" in modo
    is_chromadb = "ChromaDB" in modo
    is_legacy = "Legacy" in modo
    
    # Warings
    if is_legacy:
        st.error(
            "⚠️ **Legacy Mode (High RAM)**: Carga 3.9M de conceptos en RAM (~15GB). Precaución." 
            if lang == 'es' 
            else "⚠️ **Legacy Mode (High RAM)**: Loads 3.9M concepts into RAM (~15GB). Caution."
        )
    elif is_chromadb:
        st.success(
            "⚡ **ChromaDB Mode**: Optimizado para bajo consumo de recursos (~1GB). Requiere indexación previa." 
            if lang == 'es' 
            else "⚡ **ChromaDB Mode**: Optimized for low resource usage (~1GB). Requires prior indexing."
        )
    
    # Input section
    st.markdown("### 📝 " + ("Entrada de Texto Clínico" if lang == 'es' else "Clinical Text Input"))
    
    # Example selector
    ejemplos = EJEMPLOS_CLINICOS[lang]
    ejemplo_seleccionado = st.selectbox(
        "💡 " + ("Selecciona un ejemplo" if lang == 'es' else "Select an example"),
        [""] + [ej['nombre'] for ej in ejemplos],
        help=("Casos clínicos precargados para prueba rápida" if lang == 'es'
              else "Pre-loaded clinical cases for quick testing")
    )
    
    # Text input
    texto_default = ""
    if ejemplo_seleccionado:
        texto_default = next(ej['texto'] for ej in ejemplos if ej['nombre'] == ejemplo_seleccionado)
    
    texto_clinico = st.text_area(
        "Texto clínico" if lang == 'es' else "Clinical text",
        value=texto_default,
        height=200,
        placeholder=("Ingresa el texto de historia clínica, nota de evolución, etc." if lang == 'es'
                    else "Enter clinical history text, progress note, etc."),
        help=("Texto médico en lenguaje natural para análisis NER" if lang == 'es'
              else "Medical text in natural language for NER analysis")
    )
    
    # Process button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        procesar = st.button(
            "🔍 " + ("Procesar" if lang == 'es' else "Process"),
            type="primary",
            use_container_width=True
        )
    
    # Processing
    if procesar and texto_clinico.strip():
        try:
            # Load appropriate models
            if is_advanced:
                processor = load_advanced_models(use_chromadb=is_chromadb)
                
                msg = "Procesando con EntityLinker..." if is_legacy else "Busqueda semántica en ChromaDB..."
                if lang != 'es':
                    msg = "Processing with EntityLinker..." if is_legacy else "Semantic search in ChromaDB..."
                    
                with st.spinner("⏳ " + msg):
                    resultados = processor.procesar_avanzado(texto_clinico)
                render_advanced_results(resultados, translations)
            else:
                processor = load_basic_models()
                with st.spinner("⏳ " + ("Procesando con 3 modelos NER..." if lang == 'es' else "Processing with 3 NER models...")):
                    resultados = processor.procesar_basico(texto_clinico)
                render_basic_results(resultados, translations)
            
            # Export section
            st.markdown("---")
            st.subheader("💾 " + ("Exportar Resultados" if lang == 'es' else "Export Results"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_str = json.dumps(resultados, ensure_ascii=False, indent=2, default=str)
                st.download_button(
                    "📥 " + ("Descargar JSON" if lang == 'es' else "Download JSON"),
                    data=json_str,
                    file_name=f"ner_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV export (basic mode only)
                if not is_advanced and resultados.get('entidades_huggingface'):
                    df = pd.DataFrame(resultados['entidades_huggingface'])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 " + ("Descargar CSV" if lang == 'es' else "Download CSV"),
                        data=csv,
                        file_name=f"ner_entidades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # JSON preview
            with st.expander("👁️ " + ("Ver JSON" if lang == 'es' else "View JSON")):
                st.json(resultados)
        
        except Exception as e:
            st.error(f"❌ Error al procesar: {str(e)}")
            st.exception(e)
    
    elif procesar:
        st.warning("⚠️ " + ("Por favor ingresa texto clínico para procesar" if lang == 'es' 
                            else "Please enter clinical text to process"))
    
    # Documentation
    st.markdown("---")
    with st.expander("📚 " + ("¿Qué es esto?" if lang == 'es' else "What is this?")):
        if lang == 'es':
            st.markdown("""
            ### Named Entity Recognition (NER) Médico
            
            #### Modo Básico
            Utiliza 3 modelos especializados en paralelo:
            - **Hugging Face** (`d4data/biomedical-ner-all`): NER biomédico general
            - **SciBERT** (`en_core_sci_scibert`): Modelo científico con detección de abreviaturas
            - **BC5CDR** (`en_ner_bc5cdr_md`): Especializado en enfermedades y químicos
            
            **Detección de contexto:**
            - ⏰ Temporalidad (actual, pasado, antecedente)
            - ❌ Negación (síntomas o condiciones negadas)
            - ✓ Certeza (confirmado, probable, mencionado)
            
            #### Modo Avanzado
            Añade **EntityLinker con UMLS** (Unified Medical Language System):
            - 🗄️ ~3.9M conceptos médicos estandarizados
            - 🆔 CUI (Concept Unique Identifier)
            - 📖 Definiciones médicas completas
            - 🏷️ Clasificación por tipo semántico (TUI)
            - 🔄 Normalización y desambiguación de términos
            - 💡 Sinónimos y conceptos alternativos
            
            **Categorías automáticas:**
            - 🦠 Enfermedades
            - 🩺 Síntomas y signos
            - 💊 Medicamentos
            - 🫀 Anatomía
            - ⚕️ Procedimientos
            """)
        else:
            st.markdown("""
            ### Medical Named Entity Recognition (NER)
            
            #### Basic Mode
            Uses 3 specialized models in parallel:
            - **Hugging Face** (`d4data/biomedical-ner-all`): General biomedical NER
            - **SciBERT** (`en_core_sci_scibert`): Scientific model with abbreviation detection
            - **BC5CDR** (`en_ner_bc5cdr_md`): Specialized in diseases and chemicals
            
            **Context detection:**
            - ⏰ Temporality (current, past, history)
            - ❌ Negation (negated symptoms or conditions)
            - ✓ Certainty (confirmed, probable, mentioned)
            
            #### Advanced Mode
            Adds **EntityLinker with UMLS** (Unified Medical Language System):
            - 🗄️ ~3.9M standardized medical concepts
            - 🆔 CUI (Concept Unique Identifier)
            - 📖 Complete medical definitions
            - 🏷️ Semantic type classification (TUI)
            - 🔄 Term normalization and disambiguation
            - 💡 Synonyms and alternative concepts
            
            **Automatic categories:**
            - 🦠 Diseases
            - 🩺 Symptoms and signs
            - 💊 Medications
            - 🫀 Anatomy
            - ⚕️ Procedures
            """)
