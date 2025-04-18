from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os.path

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_pdfs(self, directory_path):
        """Load all PDFs from specified directory"""
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        # 增强文档元数据
        for doc in documents:
            if "source" in doc.metadata:
                # 提取文件名作为论文标题
                filename = os.path.basename(doc.metadata["source"])
                paper_title = os.path.splitext(filename)[0]
                
                # 添加额外的元数据
                doc.metadata["paper_title"] = paper_title
                
                # 确保页码信息可用
                if "page" not in doc.metadata:
                    doc.metadata["page"] = ""
        
        return self.text_splitter.split_documents(documents)
