from .pinecone_db import get_pinecone_index, get_vector_store

import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

PDF_DATA_DIR = "data/pdf/"


def read_pdf_file(file_path: str) -> list:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logging.info(f"FILE: db/utils.py INFO:'{file_path}' loaded")
    return documents


def chunk_data(docs: list, chunk_size=800, chunk_overlap=50) -> list:
    if not docs:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(docs)
    logging.info("FILE: db/utils.py INFO: pdf document converted to chunks")
    return split_docs


def add_documents_to_vector_store(vector_store: PineconeVectorStore, docs_to_add: list):
    if not docs_to_add:
        return
    vector_store.add_documents(docs_to_add)
    logging.info("FILE: db/utils.py INFO: Document chunks are added to Pinecone")


def delete_documents_from_vector_store(source_filepath: str):
    pinecone_index = get_pinecone_index()
    delete_response = pinecone_index.delete(filter={"source": source_filepath})
    logging.info(
        f"FILE: db/utils.py INFO: Pinecone delete response for source '{source_filepath}': {delete_response}"
    )


def delete_local_pdf(file_path: str):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        logging.info(
            f"FILE: db/utils.py INFO: File removed from local storage '{file_path}'"
        )
    else:
        logging.error(
            f"FILE: app/db/utils.py INFO: file doesn't exit int this path {file_path}"
        )


def list_local_pdfs(directory: str = PDF_DATA_DIR) -> list:
    if not os.path.isdir(directory):
        return []
    pdf_files = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(directory, f))
    ]
    return pdf_files


def get_path(file: str) -> str:
    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    file_path = os.path.join(PDF_DATA_DIR, file)
    return file_path


def existence_check(source_filepath: str) -> bool:
    vector_store = get_vector_store()
    results = vector_store.similarity_search(
        "", k=5, filter={"source": source_filepath}
    )
    return False if len(results) == 0 else True


def add_pdf(file: str):
    logging.info(f"FILE: db/utils.py INFO: Adding '{file}' to the knowledge base")
    file_path = get_path(file)
    delete_documents_from_vector_store(file_path)
    documents = read_pdf_file(file_path)
    chunks = chunk_data(documents)
    vector_store = get_vector_store()
    add_documents_to_vector_store(vector_store, chunks)


def remove_pdf(file: str):
    logging.info(f"FILE: db/utils.py INFO: Removing '{file}' from the knowledge base")
    file_path = get_path(file)
    delete_documents_from_vector_store(file_path)
    delete_local_pdf(file_path)
