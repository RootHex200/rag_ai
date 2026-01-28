"""
RAG Chain Module
Implements the Retrieval-Augmented Generation pipeline for voter queries
"""
import os
import sys
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K_RESULTS
)
from embeddings.vector_store import VoterVectorStore


class VoterRAGChain:
    """
    RAG chain for answering questions about voter information.
    """
    
    def __init__(self, vector_store: VoterVectorStore):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Initialized VoterVectorStore instance
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        self.retriever = vector_store.get_retriever(k=TOP_K_RESULTS)
        self.qa_chain = self._create_qa_chain()
        
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        
        # Custom prompt template for bilingual responses
        prompt_template = """
{system_prompt}

Context from voter database:
{context}

Question: {question}

Answer (respond in the same language as the question):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            partial_variables={"system_prompt": SYSTEM_PROMPT}
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG chain with a question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with 'answer' and 'source_documents'
        """
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def search_by_name(self, name: str, k: int = 5) -> List[Document]:
        """
        Search for voters by name.
        
        Args:
            name: Name to search for
            k: Number of results
            
        Returns:
            List of matching documents
        """
        query = f"নাম {name} name {name}"
        return self.vector_store.similarity_search(query, k=k)
    
    def search_by_father_name(self, father_name: str, k: int = 5) -> List[Document]:
        """
        Search for voters by father's name.
        
        Args:
            father_name: Father's name to search for
            k: Number of results
            
        Returns:
            List of matching documents
        """
        query = f"পিতার নাম {father_name} father {father_name}"
        return self.vector_store.similarity_search(query, k=k)
    
    def filter_by_ward(self, ward: str, k: int = 10) -> List[Document]:
        """
        Filter voters by ward.
        
        Args:
            ward: Ward number
            k: Number of results
            
        Returns:
            List of matching documents
        """
        return self.vector_store.similarity_search(
            query=f"ওয়ার্ড {ward} ward {ward}",
            k=k,
            filter_dict={"ward": ward}
        )
    
    def filter_by_occupation(self, occupation: str, k: int = 10) -> List[Document]:
        """
        Filter voters by occupation.
        
        Args:
            occupation: Occupation to filter by
            k: Number of results
            
        Returns:
            List of matching documents
        """
        return self.vector_store.similarity_search(
            query=f"পেশা {occupation} occupation {occupation}",
            k=k,
            filter_dict={"occupation": occupation}
        )
    
    def get_statistics_query(self, question: str) -> str:
        """
        Handle statistics-based queries.
        
        Args:
            question: Statistics question
            
        Returns:
            Answer string
        """
        # For statistics queries, we'll use the LLM with retrieved context
        result = self.query(question)
        return result["answer"]


class ConversationManager:
    """
    Manages conversation history and context for the chatbot.
    """
    
    def __init__(self, rag_chain: VoterRAGChain):
        """
        Initialize conversation manager.
        
        Args:
            rag_chain: VoterRAGChain instance
        """
        self.rag_chain = rag_chain
        self.history: List[Dict[str, str]] = []
    
    def add_to_history(self, question: str, answer: str):
        """Add Q&A pair to history."""
        self.history.append({
            "question": question,
            "answer": answer
        })
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    def chat(self, question: str) -> Dict[str, Any]:
        """
        Process a chat message with context awareness.
        
        Args:
            question: User's question
            
        Returns:
            Response dictionary with answer and sources
        """
        # Check if this is a follow-up question
        context_enhanced_question = question
        
        if self.history and len(self.history) > 0:
            # Add context from last exchange if needed
            last_q = self.history[-1]["question"]
            if self._is_follow_up(question):
                context_enhanced_question = f"Previous question: {last_q}\nCurrent question: {question}"
        
        # Get response from RAG chain
        result = self.rag_chain.query(context_enhanced_question)
        
        # Add to history
        self.add_to_history(question, result["answer"])
        
        return result
    
    def _is_follow_up(self, question: str) -> bool:
        """
        Detect if question is a follow-up (uses pronouns or references).
        """
        follow_up_indicators = [
            "তার", "তাদের", "তিনি", "সেই", "ঐ",  # Bengali pronouns
            "his", "her", "their", "they", "that", "those",  # English
            "কতজন", "কোথায়", "কখন",  # Bengali question words that might reference previous context
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in follow_up_indicators) and len(question.split()) < 5


def initialize_rag_system(vector_store: VoterVectorStore) -> tuple[VoterRAGChain, ConversationManager]:
    """
    Initialize the complete RAG system.
    
    Args:
        vector_store: Initialized VoterVectorStore
        
    Returns:
        Tuple of (VoterRAGChain, ConversationManager)
    """
    rag_chain = VoterRAGChain(vector_store)
    conversation_manager = ConversationManager(rag_chain)
    
    return rag_chain, conversation_manager


if __name__ == "__main__":
    # Test the RAG chain
    from utils.data_loader import load_voters_from_sql
    from embeddings.vector_store import initialize_vector_store
    from config import SQL_DUMP_PATH
    
    # Load data and initialize vector store
    print("Loading voter data...")
    voters, documents = load_voters_from_sql(SQL_DUMP_PATH)
    
    print("\nInitializing vector store...")
    vector_store = initialize_vector_store(documents)
    
    print("\nInitializing RAG chain...")
    rag_chain, conversation_manager = initialize_rag_system(vector_store)
    
    # Test queries
    test_questions = [
        "সাইফুল ইসলাম কে?",
        "১ নং ওয়ার্ডে কতজন ভোটার আছে?",
        "কৃষকদের সংখ্যা কত?",
    ]
    
    print("\n--- Testing RAG Chain ---")
    for question in test_questions:
        print(f"\nQ: {question}")
        result = conversation_manager.chat(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {len(result['source_documents'])} documents")
