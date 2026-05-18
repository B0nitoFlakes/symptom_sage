import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retrieval import retrieve, format_results
from backend.reranker import rerank

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = llm_factory("gpt-4o-mini", client=openai_client)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

test_questions = [
    # Multi-symptom combinations
    "I am experiencing a severe headache accompanied by nausea",
    "I have been experiencing abdominal pain and vomiting since last night",
    "I have a sore throat with fever and generalized body aches",
    "I am experiencing dizziness, a throbbing headache, and nausea",
    "I have been feeling fatigued constantly and experiencing throat pain when swallowing",
    "I have chest tightness accompanied by persistent coughing and shortness of breath",
    "I am experiencing stomach cramps with diarrhea and general weakness",
    "I have a headache behind my eyes with nasal congestion and facial pressure",

    # Moderate informal (realistic user language)
    "my head is really hurting and I feel like throwing up",
    "my throat is very painful and I am having difficulty swallowing",
    "I feel dizzy every time I stand up",
    "I have been coughing continuously for several days",
    "I feel extremely tired even after a full night of sleep",
    "my stomach has been hurting and I have vomited twice",

    # Severity based
    "I have a mild headache that is slightly uncomfortable",
    "I have a moderate fever and am feeling generally unwell",
    "I am experiencing severe unbearable abdominal pain",

    # Single symptom baseline
    "I have a headache",
    "I am experiencing nausea",
    "I have a sore throat"
]



def generate_answer(question: str, contexts: list) -> str:
    context_text = "\n\n".join([
        f"Condition: {r['possible_condition']}, Symptom: {r['symptom']}, Advice: {r.get('advice', '')}, When to See Doctor: {r.get('when_to_see_doctor', '')}"
        for r in contexts
    ])
    
    prompt = f"""
    You are a helpful medical assistant.

    Based on the context below, directly address the user's symptoms and provide helpful guidance.
    Only use information from the provided context.
    Do not provide a medical diagnosis.
    If information is insufficient, recommend consulting a healthcare professional.

    Context:
    {context_text}

    Question: {question}

    Provide a clear, specific answer that directly addresses the symptoms mentioned in the question.
    Answer:
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def build_dataset():
    questions = []
    answers = []
    contexts_list = []
    
    for question in test_questions:
        print(f"Processing: {question[:50]}...")
        
        results = retrieve(question, top_k=5)
        formatted = format_results(results)
        reranked = rerank(question, formatted, top_k=3)
        
        context_texts = [
            f"Condition: {r['possible_condition']}, Symptom: {r['symptom']}, Advice: {r.get('advice', '')}, When to See Doctor: {r.get('when_to_see_doctor', '')}"
            for r in reranked
        ]
        
        answer = generate_answer(question, reranked)
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(context_texts)
    
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list
    })

if __name__ == "__main__":
    # #Testing
    # question = "bro my head is killing me and i feel like throwing up"
    # results = retrieve(question, top_k=5)
    # formatted = format_results(results)
    # reranked = rerank(question, formatted, top_k=3)
    # answer = generate_answer(question, reranked)
    # print(answer)


    print("Building evaluation dataset...")
    dataset = build_dataset()
    
    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        embeddings=embeddings
    )
    
    print("\nRAGAS Evaluation Results:")
    print(results)