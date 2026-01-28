"""
Vector Store Module
Handles ChromaDB setup and embedding operations
"""
import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K_RESULTS
)


class VoterVectorStore:
    """
    Vector store for voter documents using ChromaDB and OpenAI embeddings.
    """
    
    def __init__(self):
        """Initialize the vector store with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.vector_store: Optional[Chroma] = None
        self.collection_name = COLLECTION_NAME
        self.persist_directory = CHROMA_DB_PATH
        
    def create_from_documents(self, documents: List[Dict[str, Any]]) -> Chroma:
        """
        Create a new vector store from voter documents.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            Chroma vector store instance
        """
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Convert to LangChain Document objects
        langchain_docs = []
        for doc in documents:
            langchain_doc = Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
            langchain_docs.append(langchain_doc)
        
        # Create ChromaDB vector store
        self.vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        return self.vector_store
    
    def load_existing(self) -> Optional[Chroma]:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store instance or None if not found
        """
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            return self.vector_store
        return None
    
    def get_or_create(self, documents: Optional[List[Dict[str, Any]]] = None) -> Chroma:
        """
        Get existing vector store or create new one.
        
        Args:
            documents: Documents to use if creating new store
            
        Returns:
            Chroma vector store instance
        """
        # Try to load existing
        existing = self.load_existing()
        if existing is not None:
            # Check if it has documents
            try:
                count = existing._collection.count()
                if count > 0:
                    print(f"Loaded existing vector store with {count} documents")
                    return existing
            except:
                pass
        
        # Create new if documents provided
        if documents:
            return self.create_from_documents(documents)
        
        raise ValueError("No existing vector store found and no documents provided")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if filter_dict:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
        
        return results
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search_with_score(query=query, k=k)
    
    def get_retriever(self, k: int = TOP_K_RESULTS):
        """
        Get a retriever for use with LangChain chains.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def delete_collection(self):
        """Delete the vector store collection."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print(f"Deleted vector store at {self.persist_directory}")


def initialize_vector_store(documents: Optional[List[Dict[str, Any]]] = None) -> VoterVectorStore:
    """
    Initialize and return a vector store instance.
    
    Args:
        documents: Optional documents to create store with
        
    Returns:
        Initialized VoterVectorStore instance
    """
    store = VoterVectorStore()
    store.get_or_create(documents)
    return store


if __name__ == "__main__":
    # Test the vector store
    from utils.data_loader import load_voters_from_sql
    from config import SQL_DUMP_PATH
    
    # Load documents
    voters, documents = load_voters_from_sql(SQL_DUMP_PATH)
    
    # Create vector store
    store = VoterVectorStore()
    store.create_from_documents(documents)
    
    # Test search
    print("\n--- Testing Search ---")
    results = store.similarity_search("সাইফুল ইসলাম", k=3)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200])
