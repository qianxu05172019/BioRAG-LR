from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
import os
import re

def enhance_document_metadata(documents):
    """为文档添加或增强元数据，确保所有文档都有足够的引用信息"""
    
    # 已知文献的元数据映射表 - 可以根据需要扩展
    known_papers = {
        # 使用文件名的一部分作为键，这样可以匹配不同目录下的相同文件
        "s41598-018-27829-9": {
            "paper_title": "Metabolomic profiles of bovine cumulus cells and cumulus-oocyte-complex-conditioned medium during maturation in vitro",
            "authors": "Uhde et al.",
            "journal": "Scientific Reports",
            "year": "2018",
            "volume": "8",
            "pages": "9477",
            "doi": "10.1038/s41598-018-27829-9"
        },
        # 可以添加更多已知论文
    }
    
    for doc in documents:
        source_file = doc.metadata.get('source', '').lower()
        
        # 尝试从已知论文中匹配
        matched = False
        for key, metadata in known_papers.items():
            if key in source_file:
                # 将已知元数据复制到文档元数据
                for meta_key, meta_value in metadata.items():
                    doc.metadata[meta_key] = meta_value
                matched = True
                break
        
        # 如果没有匹配到已知论文，确保至少有基本元数据
        if not matched or 'paper_title' not in doc.metadata or not doc.metadata['paper_title']:
            # 从文件名生成标题
            base_filename = os.path.basename(source_file)
            name_without_ext = os.path.splitext(base_filename)[0]
            # 清理文件名，使其更像论文标题
            clean_title = re.sub(r'[_\-]', ' ', name_without_ext).strip()
            clean_title = re.sub(r'\s+', ' ', clean_title)  # 合并多个空格
            clean_title = clean_title.title()  # 首字母大写
            
            if 'paper_title' not in doc.metadata or not doc.metadata['paper_title']:
                doc.metadata['paper_title'] = clean_title
            
            # 添加其他缺失的元数据字段
            if 'authors' not in doc.metadata:
                doc.metadata['authors'] = "Unknown Author"
            if 'journal' not in doc.metadata:
                doc.metadata['journal'] = "Unknown Journal"
    
    print("文档元数据增强完成")
    return documents

def main():
    pdf_directory = "data/papers"  
    
    # check if folder exist
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory {pdf_directory}")
        print(f"Please place your PDF files in {pdf_directory} directory and run this script again.")
        return
    
    # check if PDFs in the folder
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}!")
        print("Please add some PDF files to this directory and run this script again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    # load and process pdf
    print("Processing PDF documents...")
    document_processor = DocumentProcessor()
    documents = document_processor.load_pdfs(pdf_directory)
    
    if not documents:
        print("Error: No document chunks were generated!")
        return
    
    # 增强文档元数据
    documents = enhance_document_metadata(documents)
    
    print(f"Successfully processed {len(documents)} document chunks.")
    
    # 打印一些文档元数据样例
    if documents:
        print("\n文档元数据样例:")
        for i, doc in enumerate(documents[:3]):  # 只打印前3个文档的元数据
            print(f"Document {i+1}:")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Paper Title: {doc.metadata.get('paper_title', 'Unknown')}")
            print(f"  Authors: {doc.metadata.get('authors', 'Unknown')}")
            print(f"  Journal: {doc.metadata.get('journal', 'Unknown')}")
            print(f"  Year: {doc.metadata.get('year', 'Unknown')}")
            print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"  DOI: {doc.metadata.get('doi', 'Unknown')}")
            print(f"  Content preview: {doc.page_content[:100]}...\n")
    
    # create vector
    print("Creating vector store (this may take a while)...")
    try:
        vector_store_manager = VectorStoreManager()
        vector_store = vector_store_manager.create_vector_store(documents)
        print("Vector store created successfully!")
        print("You can now run the Streamlit app and query your documents.")
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        print("Check your OpenAI API key and ensure it's correctly set in your .env file.")

if __name__ == "__main__":
    main()