## RAG Implementations

### 1. Basic RAG Pipeline
- Manual retrieval and prompting
- FAISS + Ollama
- File: `rag_faiss_ollama.py`

### 2. LCEL-based RAG Chain
- LangChain Expression Language (LCEL)
- Modular pipeline composition
- File: `rag_chain_lcel.py`

### 3. Advanced RAG Pipeline (Optimized)
- Custom reranking logic
- Context compression
- Improved answer relevance
- File: `advanced_rag_pipeline.py`
## API Layer

This project exposes the RAG pipeline via FastAPI.

### Endpoints

- GET `/`
  - Health check

- POST `/chat`
  - Input: question
  - Output: answer generated using RAG pipeline

### Example Request
```json
{
  "question": "What is the document about?"
}
