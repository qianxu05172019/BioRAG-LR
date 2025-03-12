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
        if "citations" in message and message["citations"]:
            with st.expander("View Citations"):
                for citation in message["citations"]:
                    st.markdown(f"*{citation}*")

if prompt := st.chat_input("Ask your question about oocyte research..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if not st.session_state.rag_pipeline:
            st.error("System not initialized. Please wait...")
        else:
            with st.spinner("Researching..."):
                try:
                    # get response
                    response = st.session_state.rag_pipeline.ask(prompt)

                    if "result" in response:
                        result = response["result"]
                        answer_text = result.get("answer", "can't get answer")
                        citations = result.get("sources", [])
                    elif "answer" in response and "source_documents" in response:  
                        answer_text = response["answer"]
                        source_documents = response["source_documents"]
                        citations = [doc.metadata.get("source", "unknown source") for doc in source_documents if hasattr(doc, 'metadata')]
                    else:
                        st.error(f"Unexpected response format: {response.keys() if isinstance(response, dict) else type(response)}")
                        answer_text = "can't get answer"
                        citations = []

                    # update chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations or ["From Knowledege bank"]
                    })

                    # show response
                    st.write(answer_text)

                    # show citation
                    if citations:
                        with st.expander("View Citations"):
                            for citation in citations:
                                st.markdown(f"*{citation}*")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

with col2:
    if st.button("Export Chat"):
        st.info("Export feature coming soon!")
