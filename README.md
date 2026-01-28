# Voter Information RAG Chatbot

A bilingual (Bengali/English) chatbot that answers questions about voter information using Retrieval-Augmented Generation (RAG).

## Features

- 🤖 **AI-Powered Search**: Uses OpenAI embeddings and GPT for intelligent question answering
- 🌐 **Bilingual**: Responds in Bengali or English based on your question language
- 🔍 **Semantic Search**: Find voters by name, father's name, occupation, location, etc.
- 📊 **Statistics**: Get voter counts, demographics, and distributions
- 💬 **Chat Interface**: Beautiful Streamlit web interface with conversation history
- 🔒 **Read-Only**: Safe access to production database (works from SQL dump)

## Project Structure

```
rag_ai/
├── .env                    # API keys (gitignored)
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
├── app.py                # Streamlit web application
├── voters.sql            # Database dump file
├── utils/
│   ├── __init__.py
│   └── data_loader.py    # SQL parser and data loader
├── embeddings/
│   ├── __init__.py
│   └── vector_store.py   # ChromaDB vector store
└── rag/
    ├── __init__.py
    └── chain.py          # RAG chain implementation
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

The `.env` file has already been created with your API key.

### 3. Initialize the Database

On first run, the system will:
- Parse the `voters.sql` file
- Create embeddings for all voter records
- Store them in a local ChromaDB database

This takes ~2-3 minutes and costs approximately $0.01-0.02 in OpenAI credits.

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Sample Questions

**Bengali:**
- সাইফুল ইসলাম কে?
- মোঃ সিরাজুল মোল্যা এর ছেলে কে?
- ১ নং ওয়ার্ডে কতজন ভোটার আছে?
- কৃষকদের তালিকা দাও
- ব্যবসায়ীরা কোন এলাকায় থাকে?

**English:**
- Who is Saiful Islam?
- Who is the son of Md. Sirajul Molla?
- How many voters are in ward 1?
- List all farmers
- Where do businessmen live?

## Technical Details

### Data Source
- **Database**: PostgreSQL (Aiven Cloud)
- **Data Loading**: Parses `voters.sql` dump file (read-only, safe for production)
- **Records**: Voter registration data from Babra Union, Kalia, Narail

### AI Stack
- **LLM**: OpenAI GPT-4o-mini (cost-effective, fast)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: ChromaDB (local storage)
- **Framework**: LangChain
- **UI**: Streamlit

### Database Schema
The `voters` table contains:
- **Identity**: id, serial, name, voter_id
- **Family**: father_name, mother_name
- **Demographics**: gender, date_of_birth, occupation
- **Location**: address, ward, union, voter_area_no
- **Search**: phonetic_name, phonetic_father_name (for better Bengali search)

## Testing

### Test Data Loader
```bash
python utils/data_loader.py
```

### Test Vector Store
```bash
python embeddings/vector_store.py
```

### Test RAG Chain
```bash
python rag/chain.py
```

## Cost Estimation

### One-Time Setup
- **Embeddings**: ~3000 records × $0.00001 = $0.03

### Per Query
- **Search**: Free (local ChromaDB)
- **LLM Response**: ~$0.0001-0.0003 per query (GPT-4o-mini)

**Estimated monthly cost for 1000 queries**: ~$0.10-0.30

## Troubleshooting

### Import Errors
The LSP errors you see are normal before installing dependencies. Run:
```bash
pip install -r requirements.txt
```

### Vector Store Not Found
If the app says "No existing vector store found":
- Make sure `voters.sql` exists in the project root
- The system will automatically create embeddings on first run

### OpenAI Rate Limits
If you hit rate limits:
- The free tier has limits of 3 requests/min
- Upgrade to paid tier for higher limits
- Or add retry logic with delays

## Security Notes

- ✅ API keys stored in `.env` (gitignored)
- ✅ Read-only access to database (loads from SQL dump)
- ✅ No write operations to production database
- ✅ Local vector storage (no data sent to cloud except OpenAI API)

## Future Enhancements

- [ ] Add voice input (Bengali speech-to-text)
- [ ] Export search results to CSV/PDF
- [ ] Multi-language support (add Sylheti, Chittagonian)
- [ ] Advanced filters (age range, multiple wards)
- [ ] Voter card image generation
- [ ] Integration with live database for real-time updates

## License

This project is for educational and administrative purposes only. Voter data should be handled according to local data protection regulations.

## Support

For issues or questions, check the error messages in the Streamlit interface or console output.
