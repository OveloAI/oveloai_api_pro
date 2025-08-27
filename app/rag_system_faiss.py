import sys
import os

# --- CRITICAL FIX FOR ModuleNotFoundError ---
# Add the parent directory (project root) to Python's path.
# This ensures that 'app' is discoverable as a package when this script runs.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- END CRITICAL FIX ---


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import glob

# Import settings from our config file (now it can find 'app.config')
from app.config import settings

# Define the path to your knowledge base documents
KB_DOCS_PATH = "knowledge_base/" 

# Ensure the FAISS_DB directory exists before saving
if not os.path.exists(settings.FAISS_DB_PATH):
    os.makedirs(settings.FAISS_DB_PATH)

class OveloRAGSystem:
    """
    A RAG system that loads business documents, creates embeddings,
    and stores them in a FAISS vector database.
    """
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.HUGGINGFACE_EMBEDDING_MODEL
        )
        
    def load_documents(self) -> List[Document]:
        """Load all our knowledge documents from .txt files in the KB_DOCS_PATH."""
        documents = []
        # Update glob.glob to look inside the knowledge_base folder
        txt_files = glob.glob(os.path.join(KB_DOCS_PATH, "*.txt"))
        
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
        
        return documents
    
    def initialize_knowledge_base(self):
        """Create the vector database from our documents."""
        print("🔄 Initializing OveloAI Knowledge Base with FAISS...")
        
        documents = self.load_documents()
        
        if not documents:
            print("❌ No documents found!")
            # Exit with an error status so Render's build fails early if no documents
            raise RuntimeError("No knowledge base documents found. Cannot build FAISS index.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"📚 Processed {len(chunks)} knowledge chunks")
        
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save the index to the path specified in settings (which is "faiss_db")
        self.vector_store.save_local(settings.FAISS_DB_PATH)
        
        print(f"✅ OveloAI Knowledge Base initialized and saved to {settings.FAISS_DB_PATH}!")
        return self.vector_store
    
    def query_knowledge(self, question: str, k: int = 3):
        """Query our knowledge base for relevant documents (not used directly by API, but good for testing)."""
        if not self.vector_store:
            print("❌ Knowledge base not initialized!")
            return None
        
        try:
            results = self.vector_store.similarity_search(question, k=k)
            return results
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None

if __name__ == "__main__":
    # This block will run when rag_system_faiss.py is executed directly
    # It will initialize and save the FAISS database.
    rag_system = OveloRAGSystem()
    vector_store = rag_system.initialize_knowledge_base()

    if vector_store:
        print("\nLocal FAISS knowledge base built and saved successfully.")
        # You can add local testing logic here if needed
