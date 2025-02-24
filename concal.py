import streamlit as st
import pymongo
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MONGO_URL = st.secrets["MONGO_URL"]

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def query_llm(self, question, context):
        prompt = f"""
        You are a financial assistant who is an expert in understanding conference call transcripts. Answer the user's question based on the provided context.
        You will answer in detail with the given context. Moreover, the context contains excerpts from conversations and dialogues.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

class MongoDBClient:
    def __init__(self, uri, db_name):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
    
    def get_company_list(self):
        return [col.replace("_embeddings", "") for col in self.db.list_collection_names() if col.endswith("_embeddings")]
    
    def load_embeddings(self, company):
        collection_name = f"{company}_embeddings"
        collection = self.db[collection_name]

        embeddings = []
        texts = []

        for doc in collection.find({}, {"embedding": 1, "text": 1, "_id": 0}):
            embeddings.append(doc["embedding"])
            texts.append(doc["text"])

        if not embeddings:
            return None, None

        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        return index, texts

class AuthManager:
    def authenticate_user(self):
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False
        
        if not st.session_state["authenticated"]:
            username = st.text_input("Username", key="username")
            password = st.text_input("Password", type="password", key="password")
            if st.button("Login"):
                if username == "admin" and password == "Ishan@123":
                    st.session_state["authenticated"] = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return st.session_state["authenticated"]

def retrieve_top_chunks(index, texts, query, top_k=30):
    query_embedding = model.encode(query, convert_to_numpy=True).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    _, top_indices = index.search(query_embedding, top_k)
    from collections import OrderedDict
    unique_chunks = list(OrderedDict.fromkeys([texts[i] for i in top_indices[0]]))
    return unique_chunks

auth_manager = AuthManager()
if auth_manager.authenticate_user():
    db_client = MongoDBClient(MONGO_URL, "concal_embed")
    openai_client = OpenAIClient(OPENAI_API_KEY)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    st.title("ConCal Financial Assistant")
    company_list = db_client.get_company_list()
    selected_company = st.selectbox("Select a Company:", company_list)

    if selected_company:
        st.success(f"Company selected: {selected_company}")
        st.write("Loading embeddings...")
        index, texts = db_client.load_embeddings(selected_company)

        if index is None:
            st.error("No embeddings found for this company.")
        else:
            st.success("Embeddings loaded successfully! You can now ask a question.")
            user_question = st.text_area("Ask a question:")

            if st.button("Get Answer"):
                if user_question:
                    top_chunks = retrieve_top_chunks(index, texts, user_question, top_k=30)
                    context = "\n\n".join(top_chunks)
                    answer = openai_client.query_llm(user_question, context)
                    st.subheader("💡 Answer:")
                    st.write(answer)
                    st.subheader("📚 Top Context Used:")
                    st.write(context)
                else:
                    st.warning("Please enter a question before clicking the button.")
