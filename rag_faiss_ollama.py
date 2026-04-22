# Text → chunks → embeddings → FAISS → similarity search → LLM

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Load LLM

llm = ChatOllama(model="llama3")

# 2. Load data

loader = TextLoader("data.txt")
documents = loader.load()

# 3. split into chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

# for i, doc in enumerate(docs):
#     print(f"\n--- Chunk {i} ---\n{doc.page_content}")

#4. Embeddings

embeddings = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.from_documents(docs,embeddings)

#6. Query

def ask(query):
    #retrieve
    results = vectorstore.similarity_search(query, k=2)
    # print(results)
    context = "\n".join([doc.page_content for doc in results])
    # print(context)
    prompt = f"""
    Answer based only on this context:
    {context}
    Question : {query}
    """
    return llm.invoke(prompt).content

#Run
while True:
    q = input("You: ")
    print("AI ",ask(q))