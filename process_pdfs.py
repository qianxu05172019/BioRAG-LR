from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
import os

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
    
    print(f"Successfully processed {len(documents)} document chunks.")
    
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

