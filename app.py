import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

#creating grok client 

groq_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1"
)

#using the local embedding model

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

#connecting to chromadb client

chroma_client = chromadb.PersistentClient(
    path="chroma_storage"
)

collection = chroma_client.get_or_create_collection(
    name="document_qa_collection"
)

#loading documents from active article dictionary

def load_document_from_directory(directory_path):

    documents = []

    for file_name in os.listdir(directory_path):

        if file_name.endswith(".txt"):

            with open(
                os.path.join(directory_path, file_name),
                "r",
                encoding="utf-8"
            ) as f:

                documents.append({
                    "id": file_name,
                    "text": f.read()
                })

    return documents

#splitting into chunks

def split_text(
    text,
    chunk_size=1000,
    chunk_overlap=20
    ):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start = end - chunk_overlap

    return chunks


directory_path = "./content_dict"

documents = load_document_from_directory(
    directory_path
)

chunked_documents = []

print("=== splitting docs into chunks ===")

for doc in documents:

    chunks = split_text(doc["text"])

    for i, chunk in enumerate(chunks):

        chunked_documents.append({
            "id": f"{doc['id']}_chunk_{i}",
            "text": chunk
        })

#creating embeddings 

def get_embedding(text):

    embedding = embedding_model.encode(text)

    return embedding.tolist()


print("=== generating embeddings ===")

for doc in chunked_documents:

    doc["embedding"] = get_embedding(
        doc["text"]
    )

print("=== storing embeddings in chromadb ===")

#stoing embeddings into database

for doc in chunked_documents:

    collection.add(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )


def query_documents(question, n_results=2):
    query_embedding = get_embedding(question)
    results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("=== returning relevant chunks ===")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context = "\n\n" . join(relevant_chunks)
    prompt = f"""
    You are an assistant for question-answering tasks.

    Use the following retrieved context to answer the question.

    If the answer is not present, say:
    "I currently don't have context for that."

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=[
            {
                "role":"system",
                "content":prompt,
            },
            {
                "role":"user",
                "content":"question",
            },
        ],
    )
    answer = response.choices[0].message.content
    return answer

question = "tell me about Professor Michael S. Hart"
relevant_chunks = query_documents(question)
answer = generate_response(question,relevant_chunks)

print(answer)