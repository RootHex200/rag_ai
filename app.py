"""
Streamlit Web Application for Voter RAG Chatbot
"""
import streamlit as st
import os
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SQL_DUMP_PATH
from utils.data_loader import load_voters_from_sql, get_statistics
from embeddings.vector_store import VoterVectorStore
from rag.chain import initialize_rag_system, ConversationManager


# Page configuration
st.set_page_config(
    page_title="Voter Information Chatbot",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .voter-card {
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system (cached to avoid reloading)."""
    with st.spinner("Loading voter database..."):
        voters, documents = load_voters_from_sql(SQL_DUMP_PATH)
        stats = get_statistics(voters)
    
    with st.spinner("Initializing AI search engine..."):
        vector_store = VoterVectorStore()
        vector_store.get_or_create(documents)
    
    with st.spinner("Setting up chatbot..."):
        rag_chain, conversation_manager = initialize_rag_system(vector_store)
    
    return rag_chain, conversation_manager, voters, stats


def format_voter_card(doc) -> str:
    """Format voter information as a card."""
    metadata = doc.metadata
    
    card = f"""
<div class="voter-card">
    <h4>📋 Voter Information</h4>
    <p><strong>নাম (Name):</strong> {metadata.get('name', 'N/A')}</p>
    <p><strong>পিতার নাম (Father):</strong> {metadata.get('father_name', 'N/A')}</p>
    <p><strong>মাতার নাম (Mother):</strong> {metadata.get('mother_name', 'N/A')}</p>
    <p><strong>পেশা (Occupation):</strong> {metadata.get('occupation', 'N/A')}</p>
    <p><strong>জন্ম তারিখ (DOB):</strong> {metadata.get('date_of_birth', 'N/A')}</p>
    <p><strong>ঠিকানা (Address):</strong> {metadata.get('address', 'N/A')}</p>
    <p><strong>ওয়ার্ড (Ward):</strong> {metadata.get('ward', 'N/A')} | <strong>ইউনিয়ন (Union):</strong> {metadata.get('union', 'N/A')}</p>
    <p><strong>লিঙ্গ (Gender):</strong> {metadata.get('gender', 'N/A')}</p>
</div>
"""
    return card


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🗳️ Voter Information Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ভোটার তথ্য চ্যাটবট | Ask questions about voters in Bengali or English</div>', unsafe_allow_html=True)
    
    # Initialize system
    try:
        rag_chain, conversation_manager, voters, stats = initialize_system()
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.info("Please make sure voters.sql file exists in the project directory.")
        return
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Database Statistics")
        
        # Display statistics
        st.markdown(f"""
        <div class="stat-card">
            <h3>👥 Total Voters</h3>
            <h2>{stats['total_voters']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Occupation breakdown
        if stats['by_occupation']:
            st.subheader("👨‍💼 By Occupation")
            for occupation, count in list(stats['by_occupation'].items())[:5]:
                st.write(f"**{occupation}:** {count}")
        
        # Ward breakdown
        if stats['by_ward']:
            st.subheader("🏘️ By Ward")
            for ward, count in list(stats['by_ward'].items())[:5]:
                st.write(f"**Ward {ward}:** {count}")
        
        # Gender breakdown
        if stats['by_gender']:
            st.subheader("⚧️ By Gender")
            for gender, count in stats['by_gender'].items():
                st.write(f"**{gender}:** {count}")
        
        st.divider()
        
        # Sample queries
        st.subheader("💡 Sample Questions")
        st.markdown("""
        **Bengali:**
        - সাইফুল ইসলাম কে?
        - ১ নং ওয়ার্ডে কতজন ভোটার?
        - কৃষকদের তালিকা দাও
        - মোঃ সিরাজুল মোল্যা এর ছেলে কে?
        
        **English:**
        - Who is Saiful Islam?
        - How many voters in ward 1?
        - List all farmers
        - Who is the son of Md. Sirajul Molla?
        """)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            conversation_manager.clear_history()
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display source documents if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander(f"📚 View {len(message['sources'])} source(s)"):
                        for i, doc in enumerate(message["sources"][:3]):
                            st.markdown(format_voter_card(doc), unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about voters... (Bengali or English)"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = conversation_manager.chat(prompt)
                    answer = result["answer"]
                    sources = result["source_documents"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📚 View {len(sources)} source(s)"):
                            for i, doc in enumerate(sources[:3]):
                                st.markdown(format_voter_card(doc), unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🔒 Read-only access to voter database | Powered by OpenAI & LangChain</p>
        <p>Data source: Babra Union, Kalia, Narail | ভোটার তথ্য: বাবরা ইউনিয়ন, কালিয়া, নড়াইল</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
