from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
load_dotenv()
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index("workplace-knowledge")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=PineconeVectorStore(index=index,embedding=embeddings)

retriver=vectorstore.as_retriever(search_kwargs={"k":4})
docs = retriver.invoke(
    "How to improve posture while working?"
)


