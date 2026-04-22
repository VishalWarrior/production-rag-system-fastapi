#  What Makes “Advanced RAG”

# Basic RAG = retrieve + answer
# Advanced RAG = control + accuracy + reliability

#  4 Things You MUST Add
#  1. Reranking (VERY IMPORTANT)

# Problem:

# Retriever returns similar chunks, not best chunk

# Solution:
# 👉 Re-rank results based on relevance

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------- SETUP ---------------- #

# LLM
llm = ChatOllama(model="llama3")

# Load data
loader = TextLoader("data.txt")
documents = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
docs = splitter.split_documents(documents)

# Embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ---------------- ADVANCED LOGIC ---------------- #

# 🔥 RERANK (keyword-based simple)
def rerank(docs, query):
    scored = []
    query_words = query.lower().split()

    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query_words)
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:2]]  # top 2


# 🔥 CONTEXT COMPRESSION
def compress_docs(docs, query):
    query_words = query.lower().split()
    filtered = []

    for doc in docs:
        if any(word in doc.page_content.lower() for word in query_words):
            filtered.append(doc.page_content)

    return "\n\n".join(filtered)


# 🔥 FORMAT DOCS (WITH SOURCE TAG)
def format_docs(docs):
    return "\n\n".join(
        f"[SOURCE]\n{doc.page_content}" for doc in docs
    )


# ---------------- PROMPT ---------------- #

prompt = ChatPromptTemplate.from_template("""
You MUST answer ONLY from the context.
If answer is not found, say "I don't know".

Context:
{context}

Question:
{question}
""")


# ---------------- PIPELINE ---------------- #

def advanced_rag(query):
    # Step 1: retrieve
    retrieved_docs = retriever.invoke(query)
    # print("retrieved_docs___",retrieved_docs)

    # Step 2: rerank
    reranked_docs = rerank(retrieved_docs, query)
    # print("reranked_docs-----------------",reranked_docs)

    # Step 3: compress
    compressed_context = compress_docs(reranked_docs, query)
    # print("compressed_context--------",compressed_context)

    # Step 4: build prompt
    final_prompt = prompt.invoke({
        "context": compressed_context,
        "question": query
    })

    # Step 5: LLM
    response = llm.invoke(final_prompt)

    return response.content


# ---------------- RUN ---------------- #

# while True:
#     query = input("\nYou: ")

#     if query.lower() == "exit":
#         break

#     print("AI:", advanced_rag(query))