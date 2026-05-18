import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant = QdrantClient(host=qdrant_host, port=6333)

with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

def embed_and_store(chunks):
    collection_name = "symptom-sage"

    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("Collection created")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")

    response = client.embeddings.create(
        input = texts,
        model = "text-embedding-3-small"
    )

    embeddings = [item.embedding for item in response.data]
    print("Embeddings received!")

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload=metadatas[i]
        )
        for i in range(len(chunks))
    ]

    qdrant.upsert(collection_name, points=points)

    print(f"Successfull stored {len(chunks)} chunks in Qdrant!")
    count = qdrant.count(collection_name=collection_name)
    print(f"Collection count: {count.count}")

if __name__ == "__main__":
    try:
        embed_and_store(chunks)
    except Exception as e:
        print(f"Error: {e}")