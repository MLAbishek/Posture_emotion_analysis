from SQLAgent import sqlagent
from RAGAgent import ragagent
from langchain_core.runnables import RunnableParallel,RunnableLambda
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()


sql_runnable=RunnableLambda(sqlagent)
rag_runnable=RunnableLambda(ragagent)
parallel_runnable=RunnableParallel({
    "sql":sql_runnable,
    "rag":rag_runnable
})

llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0, 
)


def mainagent(query):
    result=parallel_runnable.invoke(query)
    sql_out=result["sql"]
    rag_out=result["rag"]
    prompt=prompt = f"""
    Question: {query}

    SQL Metrics:
    {sql_out}

    Knowledge Context:
    {rag_out}

    Combine both into one final insight.
    """
    return llm.invoke(prompt).content
print(mainagent("I am slouching too much how to eludicate it"))
