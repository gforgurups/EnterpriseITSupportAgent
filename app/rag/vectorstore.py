import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

setting = get_settings()
_embedding = None
_vectorstore=None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}

def get_embeddingmodel_dimension(model_name: str| None = None) -> int:
    if not model_name:
        raise RuntimeError("Embedding model is not configured") #generic error
    normalized_name = model_name.lower()

    if(normalized_name in EMBEDDING_DIMENSIONS):
        return EMBEDDING_DIMENSIONS[normalized_name]

    #Specific error
    raise ValueError(f"Unsupported embedding model '{model_name}' for Pinecone.")

def get_embedding_model():
    global _embedding
    if _embedding is None:
        if not setting.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing")
        _embedding = OpenAIEmbeddings(
            model=setting.embedding_model,
            api_key=setting.openai_api_key
        )
    return _embedding

def create_pinecone_index():
    dimension = get_embeddingmodel_dimension(setting.embedding_model)
    # Connect to Pinecone
    pc = Pinecone(api_key=setting.pinecone_api_key)

    # Create the index only if it does not already exist.
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if setting.pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=setting.pinecone_index_name,
            dimension=dimension, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until Pinecone reports the new index as ready.
        while not pc.describe_index(setting.pinecone_index_name).status["ready"]:
            time.sleep(1)
    
    return pc.Index(setting.pinecone_index_name)

def get_vectorstore():
    global _vectorstore
    _embedding = get_embedding_model()
    if _vectorstore is None:
        index = create_pinecone_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=_embedding,
            namespace=setting.pinecone_namespace
        )
    return _vectorstore

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k":setting.top_k})

def add_documents(chunks):
    return get_vectorstore().add_documents(chunks)