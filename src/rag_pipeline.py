from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class RAGPipeline:
    def __init__(self, persist_directory="data/chroma_db"):
        
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # Initialize memory function
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Create QA chain without initialization
        self.qa_chain = None
        self._initialize_chain()
        
    def _initialize_chain(self):
        """Initialize ConversationalRetrievalChain"""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def ask(self, query: str):
        
        try:
            
            chain_response = self.qa_chain.invoke({"question": query})  
            
            if isinstance(chain_response, dict) and "answer" in chain_response and "source_documents" in chain_response:
                answer = chain_response["answer"]
                
                source_documents = chain_response["source_documents"]
                sources = [doc.metadata.get("source", "unknown source") for doc in source_documents if hasattr(doc, 'metadata')]
                return {
                    "result": {
                        "answer": answer,
                        "sources": sources
                    }
                }

            else:
                return {
                    "result": {
                        "answer": "Error: non correct format",
                        "sources": []
                    }
                }

        except Exception as e:
            print(f"RAG error: {str(e)}")
            return {
                "result": {
                    "answer": f"mistake during checking: {str(e)}",
                    "sources": []
                }
            }

# 使用示例
if __name__ == '__main__':
    rag = RAGPipeline()
    while True:
        user_input = input("Please type your question:")
        if user_input.lower() == 'exit':
            break
        result = rag.ask(user_input)
        print("Answer：", result['result']['answer'])
