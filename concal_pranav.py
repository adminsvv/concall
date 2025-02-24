import streamlit as st
import pymongo
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Initialize OpenAI client


clientopen = OpenAI(api_key=OPENAI_API_KEY)



# MongoDB Connection
DB_URI = "mongodb+srv://pkompally:al9uD8jLFbMwYRxm@cluster0.ipw0u.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(DB_URI)
db_embeddings = client["concal_embed"]  # Database storing vector embeddings

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_company_list():
    """Retrieve the list of company names from MongoDB."""
    return [col.replace("_embeddings", "") for col in db_embeddings.list_collection_names() if col.endswith("_embeddings")]


def load_embeddings(company):
    """Loads all embeddings and text chunks for the selected company and normalizes for cosine similarity."""
    collection_name = f"{company}_embeddings"
    collection = db_embeddings[collection_name]

    embeddings = []
    texts = []

    for doc in collection.find({}, {"embedding": 1, "text": 1, "_id": 0}):
        embeddings.append(doc["embedding"])
        texts.append(doc["text"])

    if not embeddings:
        return None, None

    embeddings = np.array(embeddings, dtype="float32")

    faiss.normalize_L2(embeddings)

    # Create FAISS index with Inner Product (IP)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, texts


def retrieve_top_chunks(index, texts, query, top_k=30):
    """Retrieves the top_k most relevant chunks using FAISS with cosine similarity, ensuring uniqueness."""
    query_embedding = model.encode(query, convert_to_numpy=True).astype("float32").reshape(1, -1)

    faiss.normalize_L2(query_embedding)

    _, top_indices = index.search(query_embedding, top_k)

    from collections import OrderedDict
    unique_chunks = list(OrderedDict.fromkeys([texts[i] for i in top_indices[0]]))

    return unique_chunks


def query_llm(question, context):
    """Generates an answer using OpenAI GPT-4 with the retrieved context."""
    prompt = f"""
    You are a financial assistant who is an expert in understanding conference call transcripts. Answer the user's question based on the provided context.
    You will answer in detail with the given context. Moreover, the context contains exercepts from conversations and dialogues.
    

    Context:
    {context}

    Question: {question}

    Answer:
    """

    print("==========prompt starts==============")
    print(prompt)
    print("==========prompt ends==============")

    response = clientopen.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


st.title("ConCal Financial Assistant")

company_list = get_company_list()
selected_company = st.selectbox("Select a Company:", company_list)

if selected_company:
    st.success(f"Company selected: {selected_company}")

    st.write("Loading embeddings...")
    index, texts = load_embeddings(selected_company)

    if index is None:
        st.error("No embeddings found for this company.")
    else:
        st.success("Embeddings loaded successfully! You can now ask a question.")

        user_question = st.text_area("Ask a  question:")

        if st.button("Get Answer"):
            if user_question:

                top_chunks = retrieve_top_chunks(index, texts, user_question, top_k=30)


                context = "\n\n".join(top_chunks)


                answer = query_llm(user_question, context)


                st.subheader("💡 Answer:")
                st.write(answer)

                st.subheader("📚 Top Context Used:")
                st.write(context)
            else:
                st.warning("Please enter a question before clicking the button.")
