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

llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0, 
)

def ragagent(query):
    docs=retriver.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt=prompt = """
    You are a helpful assistant.
    Use the following context to answer the question.
    Context: {context}
    Question: {query}
    Answer: 
    """.format(context=context,query=query)
    return llm.invoke(prompt).content




