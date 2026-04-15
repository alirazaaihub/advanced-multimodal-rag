from typing_extensions import TypedDict
from collections import defaultdict
from typing import List
from pydantic import BaseModel
import os

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.memory import ConversationSummaryBufferMemory, ConversationEntityMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


# CONFIG
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)


# MEMORY
short_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,
    return_messages=True
)

entity_memory = ConversationEntityMemory(llm=llm)


# VECTOR STORE

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = Chroma(
    persist_directory="./memory_db",
    embedding_function=embeddings
)

# RERANKER


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# STATE


class ChatState(TypedDict):
    user_input: str
    expanded_queries: List[str]
    retrieved_docs: List[Document]
    final_docs: List[Document]
    response: str


# STEP 1: QUERY EXPANSION


class QueryStruct(BaseModel)
    queries: List[str]

s_llm = llm.with_structured_output(QueryStruct)

def expand_query(state):
    query = state["user_input"]

    result = s_llm.invoke(f"Generate 3 variations of: {query}")
    return {"expanded_queries": result.queries}


# STEP 2: HYBRID RETRIEVAL


def rrf_fusion(results: List[List[Document]], k=60):
    scores = defaultdict(float)
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            scores[key] += 1 / (k + rank)
            doc_map[key] = doc

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in sorted_docs]

def hybrid_retrieve(query):
    # Dense retrieval
    vector_docs = vectorstore.similarity_search(query, k=10)

    # Sparse retrieval
    raw_docs = vectorstore.get().get("documents", [])
    all_docs = [Document(page_content=d) for d in raw_docs]

    if len(all_docs) == 0:
        return vector_docs

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 10
    bm25_docs = bm25.invoke(query)

    return rrf_fusion([vector_docs, bm25_docs])

def retrieve(state):
    all_results = []

    for q in state["expanded_queries"]:
        docs = hybrid_retrieve(q)
        all_results.append(docs)

    fused = rrf_fusion(all_results)

    return {"retrieved_docs": fused[:10]}


# STEP 3: RERANK


def rerank(state):
    query = state["user_input"]
    docs = state["retrieved_docs"]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return {"final_docs": [doc for doc, _ in ranked[:5]]}

# STEP 4: GENERATE

def generate(state):
    short_hist = short_memory.load_memory_variables({}).get("history", "")
    entity_hist = entity_memory.load_memory_variables({}).get("entities", "")

    context = "\n".join([d.page_content for d in state["final_docs"]])

    prompt = f"""
Conversation Summary:
{short_hist}

Known Entities:
{entity_hist}

Retrieved Context:
{context}

User Question:
{state['user_input']}
"""

    result = llm.invoke(prompt)

    # Save memory
    short_memory.save_context(
        {"input": state["user_input"]},
        {"output": result.content}
    )

    entity_memory.save_context(
        {"input": state["user_input"]},
        {"output": result.content}
    )

    # Long-term memory (vector)
    vectorstore.add_texts([f"User: {state['user_input']}\nBot: {result.content}"])

    return {"response": result.content}


# GRAPH
builder = StateGraph(ChatState)

builder.add_node("expand", expand_query)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate", generate)

builder.add_edge(START, "expand")
builder.add_edge("expand", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# RUN


if __name__ == "__main__":
    print("🚀 Advanced Memory + RAG Chatbot Started")

    while True:
        user = input("User: ")

        if user.lower() == "exit":
            break

        result = graph.invoke({"user_input": user})

        print("Bot:", result["response"])
