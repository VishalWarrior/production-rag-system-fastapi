from fastapi import FastAPI
from pydantic import BaseModel

from rag3 import advanced_rag

app  = FastAPI()

#Request schema

class QueryRequest(BaseModel):
    question : str

# health check
@app.get("/")
def read_root():
    return {"message": "RAG API is running"}
@app.post("/chat")
def chat(request: QueryRequest):
    answer = advanced_rag(request.question)
    return {
        "question":request.question,
        "answer":answer
    }
