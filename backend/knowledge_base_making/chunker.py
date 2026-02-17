from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def getChunk(filename,source=None,topic=None):
    loader = PyPDFLoader(filename)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_documents(documents)
    if(source!=None and topic !=None):
       
        for c in chunks:
            c.metadata = {
                "source": source,
                "topic": topic
            }
    return chunks



pdfs = [
    ("Ergonomics.pdf", "ergonomics", "posture"),
    ("Fatigue.pdf", "fatigue", "drowsiness"),
    ("Workplace_enhancement.pdf", "workplace", "enhancement"),
    ("Productivity.pdf", "productivity", "focus")
]
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.create_index(
    name="workplace-knowledge",
    dimension=384,   # MiniLM dimension
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
index = pc.Index("workplace-knowledge") 
for file,source,topic in pdfs:

    chunk=getChunk(file,source=source,topic=topic)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )
    vector_store.add_documents(chunk) 
print("Done")


