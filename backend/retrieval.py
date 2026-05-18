import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant = QdrantClient(host=qdrant_host, port=6333)
COLLECTION_NAME="symptom-sage"

def embed_query(query: str):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

def retrieve(query:str, top_k: int=5):
    query_vector = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    return results.points

def format_results(results):
    formatted = []
    for result in results:
       formatted.append({
            "score": round(result.score, 4),
            "symptom": result.payload["symptom"],
            "possible_condition": result.payload["possible_condition"],
            "advice": result.payload["advice"],
            "when_to_see_doctor": result.payload["when_to_see_doctor"],
            "source": result.payload["source"],
            "url": result.payload["url"]
       })
    
    return formatted

if __name__ == "__main__":
    query = "I have terrible headache and feel nauseous"
    print(f"Query: {query}\n")

    results = retrieve(query)
    formatted = format_results(results)

    for r in formatted:
        print(f"Score: {r['score']}")
        print(f"Symptom: {r['symptom']}")
        print(f"Condition: {r['possible_condition']}")
        print(f"Condition: {r['advice']}")
        print(f"Condition: {r['when_to_see_doctor']}")
        print(f"Source: {r['source']}")
        print(f"URL: {r['url']}")
        print("---")
