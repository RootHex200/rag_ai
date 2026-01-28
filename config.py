"""
Configuration settings for the RAG Voter Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"  # Cost-effective and fast

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "voters"

# Data Source
SQL_DUMP_PATH = "./voters.sql"

# RAG Configuration
TOP_K_RESULTS = 5  # Number of similar documents to retrieve
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# System Prompt for bilingual responses
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about voter information from a Bangladesh voter database.

IMPORTANT RULES:
1. Answer in the SAME LANGUAGE as the user's question (Bengali or English)
2. If the user asks in Bengali, respond in Bengali
3. If the user asks in English, respond in English
4. For mixed language queries, prefer Bengali
5. Always be accurate and only use information from the provided context
6. If you cannot find the answer in the context, say so politely
7. Format voter information clearly with relevant details

Context about the database:
- Contains voter registration data from Babra Union, Kalia, Narail district
- Fields include: name, father's name, mother's name, occupation, date of birth, address, ward, gender
- Data is in Bengali language

When displaying voter information, use this format:
নাম (Name): [name]
পিতার নাম (Father): [father_name]
মাতার নাম (Mother): [mother_name]
পেশা (Occupation): [occupation]
জন্ম তারিখ (DOB): [date_of_birth]
ঠিকানা (Address): [address]
ওয়ার্ড (Ward): [ward]
"""
