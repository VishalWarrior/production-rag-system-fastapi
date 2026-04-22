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
 
  - ## API Testing

### Swagger UI
FastAPI provides built-in interactive API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

You can test endpoints directly from the browser.

---

### Example API Requests

#### Health Check
```bash
curl -X GET http://127.0.0.1:8000/

### Example Request
```json
{
  "question": "What is the document about?"
}
