from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
db=SQLDatabase.from_uri(os.environ["DATABASE_URL"])

system_prompt = """

You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.



""".format(
    dialect=db.dialect,
    top_k=5
)
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0, 
)

toolkit = SQLDatabaseToolkit(db=db,llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    system_prompt=system_prompt,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)
def sqlagent(query):
    return agent.invoke({"input": query})