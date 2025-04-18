from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
import os.path


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
        """Initialize ConversationalRetrievalChain with custom prompt"""
        # 自定义提示模板，要求模型添加引用标记
        qa_prompt = ChatPromptTemplate.from_template("""
        你是一个专门回答有关卵子成熟的研究问题的专家。
        利用下面提供的上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        
        对于回答中的每一个事实或观点，请在适当的位置使用引用标记 [1], [2] 等来引用信息的来源。
        
        上下文：
        {context}
        
        聊天历史：
        {chat_history}
        
        问题：{question}
        """)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True
        )

    def _format_citation(self, doc):
        """Format citation to include more detailed information"""
        if not hasattr(doc, 'metadata'):
            return "Unknown source"
            
        metadata = doc.metadata
        source = metadata.get("source", "unknown source")
        page_number = metadata.get("page", "")
        paper_title = metadata.get("paper_title", "")
        
        # 如果有paper_title就直接使用
        if paper_title:
            citation = f"{paper_title}"
        # 否则从source提取文件名
        elif source and isinstance(source, str):
            filename = os.path.basename(source)
            paper_title = os.path.splitext(filename)[0]
            citation = f"{paper_title}"
        else:
            return source
            
        # 添加页码信息
        if page_number:
            citation += f" (Page {page_number})"
                
        return citation

    def ask(self, query: str):
        
        try:
            
            chain_response = self.qa_chain.invoke({"question": query})  
            
            if isinstance(chain_response, dict) and "answer" in chain_response and "source_documents" in chain_response:
                answer = chain_response["answer"]
                
                source_documents = chain_response["source_documents"]
                # 增强的引用格式
                sources = [self._format_citation(doc) for doc in source_documents]
                
                # 创建引用编号的映射
                unique_sources = list(set(sources))
                
                # 查看已经存在于答案中的引用标记 [1], [2] 等
                citation_links = []
                for i, source in enumerate(unique_sources):
                    citation_idx = i + 1
                    if f"[{citation_idx}]" in answer:
                        citation_links.append(f"[{citation_idx}] {source}")
                
                # 如果没有任何引用标记，就在末尾添加一个通用引用
                if not citation_links and unique_sources:
                    answer += " [1]"
                    citation_links = [f"[1] {', '.join(unique_sources)}"]
                
                return {
                    "result": {
                        "answer": answer,
                        "sources": citation_links
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
        print("Sources：")
        for source in result['result']['sources']:
            print(f"  {source}")
