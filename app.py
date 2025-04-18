import streamlit as st
from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
from src.rag_pipeline import RAGPipeline
import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY or provide in .env file")
st.set_page_config(
    page_title="Oocyte Expert",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
# chat_history: stores the conversation between user and assistant
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# rag_pipeline: stores the RAG pipeline instance for question answering
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

# is_initialized: tracks whether the knowledge base is loaded and ready
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False

# vector_store: holds the vector database containing embedded documents
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("🧬 Oocyte Expert")
    st.markdown("### About\nThis AI assistant specializes in oocyte maturation research.")
    
    if st.session_state.is_initialized:
        st.success("Knowledge Base: Active ✅")
    else:
        st.warning("Knowledge Base: Loading...")

    if st.button("Reset System"):
        st.session_state.chat_history = []
        st.experimental_rerun()

st.title("Oocyte Research Assistant")

# Initialize 
try:
    vector_db_path = "data/chroma_db"  
    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vector_store(vector_db_path)
    st.session_state.vector_store = vector_store
except ValueError as e:
    st.error(f"Vector store not found. Error: {str(e)}")
    st.stop()

# Initialize RAGPipeline
if not st.session_state.is_initialized:
    with st.spinner("Initializing knowledge base..."):
        try:
            st.session_state.rag_pipeline = RAGPipeline("data/chroma_db")
            st.session_state.is_initialized = True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()

# show chat history
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # 修改：总是显示引用部分，即使是"No source documents found"
        if "citations" in message and message["citations"]:
            with st.expander("📚 References", expanded=False):
                for citation in message["citations"]:
                    st.markdown(citation)

if prompt := st.chat_input("Ask your question about oocyte research..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if not st.session_state.rag_pipeline:
            st.error("System not initialized. Please wait...")
        else:
            with st.spinner("Researching..."):
                try:
                    # Get response from RAG pipeline
                    response = st.session_state.rag_pipeline.ask(prompt)
                    
                    # Debug information
                    print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
                    
                    # Extract answer and citations
                    if isinstance(response, dict):
                        answer_text = response.get("answer", "Could not generate an answer")
                        
                        # Try to get citations from different possible response formats
                        if "formatted_citations" in response:
                            formatted_citations = response["formatted_citations"]
                        elif "source_documents" in response:
                            # Format source documents into citations
                            source_docs = response["source_documents"]
                            formatted_citations = []
                            for doc in source_docs:
                                if hasattr(doc, 'metadata'):
                                    source = doc.metadata.get('source', 'Unknown source')
                                    paper_title = doc.metadata.get('paper_title', '')
                                    page = doc.metadata.get('page', '')
                                    
                                    citation = f"**Source**: {source}"
                                    if paper_title:
                                        citation += f" | **Title**: {paper_title}"
                                    if page:
                                        citation += f" | **Page**: {page}"
                                    
                                    formatted_citations.append(citation)
                            
                            if not formatted_citations:
                                formatted_citations = ["No source documents found"]
                        else:
                            formatted_citations = ["No source information available"]
                    else:
                        answer_text = str(response)
                        formatted_citations = ["No source information available"]
                    
                    # Update chat history with answer and citations
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": formatted_citations
                    })
                    
                    # Display answer
                    st.write(answer_text)
                    
                    # 修改：总是显示引用部分，即使是"No source documents found"
                    with st.expander("📚 References", expanded=False):
                        for citation in formatted_citations:
                            st.markdown(citation)
                                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

with col2:
    if st.button("Export Chat"):
        # Create a formatted chat export
        chat_export = "# Oocyte Research Chat Export\n\n"
        for msg in st.session_state.chat_history:
            role = "🧑‍💼 User" if msg["role"] == "user" else "🤖 Assistant"
            chat_export += f"## {role}\n\n{msg['content']}\n\n"
            # 修改：总是包含引用，即使是"No source documents found"
            if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
                chat_export += "### References\n\n"
                for citation in msg["citations"]:
                    chat_export += f"- {citation}\n"
                chat_export += "\n---\n\n"
        
        # Download button for the chat export
        st.download_button(
            label="Download Chat",
            data=chat_export,
            file_name="oocyte_research_chat.md",
            mime="text/markdown"
        )