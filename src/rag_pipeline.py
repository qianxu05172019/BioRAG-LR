from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from src.embeddings import VectorStoreManager
import os
import re

class RAGPipeline:
    def __init__(self, vector_db_path):
        """
        Initialize the RAG pipeline with a vector database
        
        Args:
            vector_db_path: Path to the Chroma vector database
        """
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = self.vector_store_manager.load_vector_store(vector_db_path)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
        )
        
        # Define a better prompt template for our RAG pipeline
        self.template = """
        You are an expert AI assistant specializing in oocyte maturation research. 
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always format your answer in a clear, scientific manner.
        
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        self.QA_PROMPT = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        
        self.llm = OpenAI(
            temperature=0,  # More deterministic responses
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize the QA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" means to stuff all documents into the prompt
            retriever=self.retriever,
            return_source_documents=True,  # Important for citations
            chain_type_kwargs={"prompt": self.QA_PROMPT}
        )
    
    def format_sources(self, source_documents):
        """
        Format source documents into a standardized citation format
        
        Args:
            source_documents: List of source documents returned by the retriever
            
        Returns:
            List of formatted citations
        """
        citations = []
        seen_sources = set()  # To avoid duplicate citations
        
        # Debug output
        print(f"Number of source documents retrieved: {len(source_documents)}")
        
        for i, doc in enumerate(source_documents[:2]):  # Print first 2 docs for debugging
            if hasattr(doc, 'metadata'):
                print(f"Document {i+1} metadata: {doc.metadata}")
            else:
                print(f"Document {i+1} has no metadata attribute")
            
            if hasattr(doc, 'page_content'):
                print(f"Document {i+1} content preview: {doc.page_content[:100]}...")
            else:
                print(f"Document {i+1} has no page_content attribute")
        
        for i, doc in enumerate(source_documents):
            # 确保文档有元数据
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            
            # 获取基本引用信息，使用合理的默认值
            paper_title = doc.metadata.get('paper_title', '')
            authors = doc.metadata.get('authors', '')
            journal = doc.metadata.get('journal', '')
            year = doc.metadata.get('year', '')
            volume = doc.metadata.get('volume', '')
            pages = doc.metadata.get('page', doc.metadata.get('pages', ''))
            doi = doc.metadata.get('doi', '')
            source = doc.metadata.get('source', '')
            
            # 如果没有标题，尝试从文件名生成
            if not paper_title and source:
                base_filename = os.path.basename(source)
                name_without_ext = os.path.splitext(base_filename)[0]
                paper_title = re.sub(r'[_\-]', ' ', name_without_ext).title()
            
            # 构建引用文本
            citation_parts = []
            
            # 添加标题（必须有）
            if paper_title:
                citation_parts.append(f"**Title**: {paper_title}")
            else:
                citation_parts.append(f"**Document {i+1}**")
            
            # 添加其他可选元数据
            if authors:
                citation_parts.append(f"**Authors**: {authors}")
            
            if journal:
                journal_info = journal
                if volume:
                    journal_info += f" {volume}"
                if pages:
                    journal_info += f", {pages}"
                if year:
                    journal_info += f" ({year})"
                citation_parts.append(f"**Journal**: {journal_info}")
            elif year:
                citation_parts.append(f"**Year**: {year}")
            
            if doi:
                citation_parts.append(f"**DOI**: {doi}")
            
            # 构建完整引用
            citation = " | ".join(citation_parts)
            
            # 创建引用键以避免重复
            if source:
                citation_key = source
            else:
                citation_key = f"{paper_title}_{authors}_{journal}"
            
            if citation_key not in seen_sources:
                citations.append(citation)
                seen_sources.add(citation_key)
        
        if not citations and len(source_documents) > 0:
            # 找到文档但没有足够的元数据创建引用
            return ["Sources found but citation metadata is incomplete"]
        elif not citations:
            # 没有找到相关文档
            return ["No relevant sources found"]
        
        return citations
    
    def ask(self, question):
        """
        Ask a question to the RAG pipeline
        
        Args:
            question: The question to ask
            
        Returns:
            Dict containing the answer and formatted citations
        """
        try:
            # Get response from the QA chain
            response = self.qa({"query": question})
            
            # Debug info
            print(f"RAG response keys: {response.keys()}")
            source_docs = response.get("source_documents", [])
            print(f"Number of source documents in response: {len(source_docs)}")
            
            # Format the source documents
            formatted_citations = self.format_sources(source_docs)
            
            # Return a structured response
            return {
                "answer": response.get("result", "Could not generate an answer"),
                "source_documents": source_docs,
                "formatted_citations": formatted_citations
            }
            
        except Exception as e:
            import traceback
            print(f"Error in RAG pipeline: {str(e)}")
            print(traceback.format_exc())
            
            # Return a fallback response
            return {
                "answer": f"I encountered an error while processing your question. Please try again or rephrase your question. Error: {str(e)}",
                "source_documents": [],
                "formatted_citations": ["Error processing sources"]
            }