import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, Index


load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")


def get_embeddings_client() -> CohereEmbeddings:
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY
    )
    return embeddings


def get_pinecone_index() -> Index:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    return index


def get_vector_store() -> PineconeVectorStore:
    embeddings = get_embeddings_client()
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    return vector_store
