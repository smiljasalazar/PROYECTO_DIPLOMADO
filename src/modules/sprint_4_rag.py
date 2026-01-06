"""
Sprint 4: RAG Clínico
Sistema de consulta con documentos y base de conocimiento UMLS.
Modos: LITE (solo docs) y COMPLETE (UMLS + docs)
"""

import streamlit as st
import os

# Lazy imports
_rag = None
_doc_loader = None


def get_rag(mode: str = "lite"):
    """Get RAG processor (lazy load)."""
    global _rag
    
    # Check if mode changed
    if _rag is not None and _rag.mode.value != mode:
        _rag = None
    
    if _rag is None:
        with st.spinner(f"⏳ Iniciando RAG ({mode.upper()})..."):
            from utils.rag_processor import RAGProcessor, RAGMode, LLMType
            
            rag_mode = RAGMode.COMPLETE if mode == "complete" else RAGMode.LITE
            
            _rag = RAGProcessor(
                mode=rag_mode,
                default_llm=LLMType.OLLAMA_LLAMA2
            )
    
    return _rag


def render_sidebar(lang: str) -> dict:
    """Render sidebar configuration."""
    config = {}
    
    st.sidebar.markdown("### ⚙️ Configuración")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Modo RAG:",
        options=["lite", "complete"],
        format_func=lambda x: "🚀 Lite (rápido)" if x == "lite" else "📚 Completo (UMLS)",
        index=0,
        help="Lite: Solo documentos subidos (rápido)\nCompleto: UMLS + documentos (lento)"
    )
    config["mode"] = mode
    
    st.sidebar.divider()
    
    # LLM selection - Solo opciones funcionales
    llm = st.sidebar.radio(
        "Modelo LLM:",
        options=["search_only", "ollama_llama2"],
        format_func=lambda x: {
            "search_only": "🔍 Solo Búsqueda (sin LLM)",
            "ollama_llama2": "🦙 Llama2 (Ollama)",
        }[x],
        index=1,  # Default to Llama2
        help="Ollama: Requiere que 'ollama serve' esté corriendo"
    )
    config["llm"] = llm
    
    st.sidebar.divider()
    
    # Document upload
    st.sidebar.markdown("### 📄 Subir Documentos")
    
    uploaded = st.sidebar.file_uploader(
        "PDF o TXT:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded:
        if st.sidebar.button("📥 Indexar", use_container_width=True):
            process_uploads(uploaded, mode)
    
    # Stats
    rag = get_rag(mode)
    stats = rag.get_stats()
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Estadísticas")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("📄 Docs", stats["documents"])
    col2.metric("📚 UMLS", f"{stats['umls']:,}" if stats['umls'] else "N/A")
    
    # Clear button
    if stats["documents"] > 0:
        if st.sidebar.button("🗑️ Limpiar documentos", use_container_width=True):
            rag.clear_documents()
            st.rerun()
    
    return config


def process_uploads(files, mode: str):
    """Process uploaded files."""
    rag = get_rag(mode)
    
    for file in files:
        content = ""
        
        if file.name.endswith(".txt"):
            content = file.read().decode("utf-8")
        elif file.name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(file.read()))
                content = "\n".join([page.extract_text() for page in reader.pages])
            except Exception as e:
                st.sidebar.error(f"Error PDF: {e}")
                continue
        
        if content:
            # Chunk the content
            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
            
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    rag.add_document(chunk, {"source": file.name})
            
            st.sidebar.success(f"✅ {file.name}: {len(chunks)} chunks")
    
    st.rerun()


def render_chat(config: dict, lang: str):
    """Render chat interface."""
    
    # Initialize chat history
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = []
    
    # Display chat
    for msg in st.session_state.rag_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg.get("sources"):
                with st.expander("📚 Fuentes"):
                    for i, src in enumerate(msg["sources"][:3], 1):
                        st.markdown(f"**[{i}] {src.get('source', 'Doc')}**")
                        st.caption(src.get("content", "")[:150] + "...")
    
    # Chat input
    if prompt := st.chat_input("Escribe tu pregunta médica..."):
        # Add user message
        st.session_state.rag_chat.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Buscando..."):
                rag = get_rag(config["mode"])
                
                # Search only mode - no LLM
                if config["llm"] == "search_only":
                    sources = rag.search(prompt)
                    
                    if sources:
                        answer = "**📚 Fragmentos relevantes encontrados:**\n\n"
                        for i, src in enumerate(sources[:5], 1):
                            answer += f"**[{i}]** {src.get('content', '')[:300]}...\n\n"
                    else:
                        answer = "No se encontraron documentos relevantes."
                    
                    st.markdown(answer)
                    
                    st.session_state.rag_chat.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    # Ollama mode
                    from utils.rag_processor import LLMType
                    llm_map = {
                        "ollama_llama2": LLMType.OLLAMA_LLAMA2,
                    }
                    
                    result = rag.query(prompt, llm_type=llm_map[config["llm"]])
                    
                    st.markdown(result.answer)
                    
                    if result.sources:
                        with st.expander("📚 Fuentes"):
                            for i, src in enumerate(result.sources[:3], 1):
                                st.markdown(f"**[{i}] {src.get('source', 'Doc')}**")
                                st.caption(src.get("content", "")[:150] + "...")
                    
                    st.session_state.rag_chat.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources
                    })
    
    # Clear chat
    if st.session_state.rag_chat:
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.rag_chat = []
            st.rerun()


def render(translations: dict, lang: str):
    """Main render function."""
    
    # Description
    st.markdown("""
    Sistema RAG para consultas médicas sobre documentos clínicos.
    Sube guías, protocolos o artículos para hacer preguntas.
    """)
    
    # Mode explanation
    with st.expander("ℹ️ ¿Qué modo usar?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🚀 Modo LITE** (Recomendado)
            - Solo busca en documentos subidos
            - Carga rápida (~2 seg)
            - Bajo consumo RAM
            """)
        with col2:
            st.markdown("""
            **📚 Modo COMPLETO**
            - Busca en UMLS + documentos
            - Requiere ChromaDB UMLS
            - Mayor uso de RAM
            """)
    
    st.divider()
    
    # Sidebar config
    config = render_sidebar(lang)
    
    # Warning for no documents
    rag = get_rag(config["mode"])
    if rag.docs_collection.count() == 0:
        st.warning("⚠️ No hay documentos cargados. Sube PDFs o TXTs en el panel izquierdo.")
    
    # Chat
    render_chat(config, lang)
    
    # Ethics
    st.divider()
    with st.expander("⚠️ Consideraciones Éticas"):
        st.markdown("""
        - 🏥 Sistema **experimental**, no reemplaza consejo médico
        - 📖 Las respuestas se basan SOLO en documentos cargados
        - 🔒 No ingresar datos reales de pacientes
        """)
    
    st.caption("📓 Notebook: `notebooks/04_rag_biomistral.ipynb`")
