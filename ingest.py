import json
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

load_dotenv()


def partition_document(file_path: str):
    print("Partitioning PDF...")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    print("Elements extracted:", len(elements))
    return elements


def create_chunks(elements):

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )

    print("Chunks created:", len(chunks))
    return chunks


def separate_content_types(chunk):

    content = {
        "text": chunk.text,
        "tables": [],
        "images": []
    }

    if hasattr(chunk.metadata, "orig_elements"):

        for el in chunk.metadata.orig_elements:

            if el.category == "Table":
                table_html = getattr(el.metadata, "text_as_html", el.text)
                content["tables"].append(table_html)

            if el.category == "Image":
                if hasattr(el.metadata, "image_base64"):
                    content["images"].append(el.metadata.image_base64)

    return content


from langchain_core.messages import HumanMessage

def ai_summary(text: str, tables: list, images: list):
    GROQ_API_KEY = "Enter your api key"

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = f"Create a searchable description for document retrieval.\n\nTEXT:\n{text}"

    if tables:
        prompt += "\nTABLES:\n"
        for t in tables:
            prompt += t + "\n"

    if images:
        prompt += "\nIMAGES:\n"
        for idx, img in enumerate(images):
            # You can just mention that images exist, base64 embedding optional
            prompt += f"[Image {idx + 1} attached]\n"

    # Pass the full prompt as a string
    msg = HumanMessage(content=prompt)

    response = llm.invoke([msg])

    return response.content


def process_chunks(chunks):

    docs = []

    for i, chunk in enumerate(chunks):

        print("Processing chunk", i + 1)

        content = separate_content_types(chunk)

        if content["tables"] or content["images"]:
            text = ai_summary(
                content["text"],
                content["tables"],
                content["images"]
            )
        else:
            text = content["text"]

        doc = Document(
            page_content=text,
            metadata={
                "original_content": json.dumps(content)
            }
        )

        docs.append(doc)

    return docs


def create_vector_store(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Vector DB created")

    return db


def run_pipeline(pdf_path):

    elements = partition_document(pdf_path)

    chunks = create_chunks(elements)

    processed = process_chunks(chunks)

    create_vector_store(processed)


if __name__ == "__main__":

    run_pipeline("Enter pdf path")
