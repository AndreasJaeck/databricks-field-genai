# Databricks notebook source
# MAGIC %md # Use Agent's to interact with a SQL database 
# MAGIC
# MAGIC The following code showcases an example of the Databricks SQL Agent. With the Databricks SQL agent any Databricks users can interact with a specified schema in Databrick Unity Catalog and generate insights on their data.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ![Agent](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*3O1QAj_62FzWydQj.png)

# COMMAND ----------

# MAGIC %md ### Imports
# MAGIC
# MAGIC Databricks recommends the latest version of `langchain` and the `databricks-sql-connector`.

# COMMAND ----------

# MAGIC %pip install --upgrade langchain databricks-sql-connector sqlalchemy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### SQL Database Agent
# MAGIC
# MAGIC This is an example of how to interact with a certain schema in Unity Catalog. Please note that the agent can't create new tables or delete tables. It can only query tables.
# MAGIC
# MAGIC The database instance is created within:
# MAGIC ```
# MAGIC db = SQLDatabase.from_databricks(catalog="...", schema="...")
# MAGIC ```
# MAGIC And the agent (and the required tools) are created by:
# MAGIC ```
# MAGIC toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# MAGIC agent = create_sql_agent(llm=llm, toolkit=toolkit, **kwargs)
# MAGIC ```

# COMMAND ----------

# Get access token of current user
DATABRICKS_TOKEN = dbutils.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create SQL Agent 
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain import OpenAI

# Define the Databricks db
db = SQLDatabase.from_databricks(catalog="samples", schema="nyctaxi")
#db = SQLDatabase.from_databricks(catalog="alan_demos", schema="events_demo")

# Use OpenAI client to connect with databricks llm endpoints
llm = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints",
    model="databricks-meta-llama-3-70b-instruct",
)

# Create SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create SQL agent
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Run SQL Agent

# COMMAND ----------

agent.run("How many events have happend for each platform type? Please use events_gold table.")

# COMMAND ----------

agent.run("What is the longest trip distance and how long did it take?")
