# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Setup

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Install Dependencies 

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1  cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade SQLAlchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#%pip install databricks-vectorsearch

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Set variables and permissions, create db's

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    indexes = vsc.list_indexes(endpoint_name).get("vector_indexes", list())
    if any(index_full_name == index.get("name") for index in indexes):
      return True
    #Temp fix when index is not available in the list
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready')
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False

# COMMAND ----------

def display_chat(chat_history, response):
  def user_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #c2efff; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px;">
        {message}
      </div>"""
  def assistant_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #e3f6fc; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
        <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>
        {message}
      </div>"""
  chat_history_html = "".join([user_message_html(m["content"]) if m["role"] == "user" else assistant_message_html(m["content"]) for m in chat_history])
  answer = response["result"].replace('\n', '<br/>')
  response_html = f"""{answer}"""

  displayHTML(chat_history_html + assistant_message_html(response_html))

# COMMAND ----------

import os

# Set configuration

# Catalogs/ DB's / Tables
# ---------------------------
# Name of the catalog
catalog = "dbdemos_aj"
# Name of the Schema / Database
schema = "allianz_chat"
# Name of the volumes
raw_data_volume ="raw_data"
checkpoints_volume ="checkpoints"

# Path/Name of document table 
document_table_name = "documents"
document_table = f"{catalog}.{schema}.{document_table_name}"

# Path/Name of parent table 
parent_document_table_name = f"{document_table_name}_parent_split"
parent_document_table = f"{catalog}.{schema}.{parent_document_table_name}"

# Path/Name of child document table 
child_document_table_name = f"{document_table_name}_child_split"
child_document_table = f"{catalog}.{schema}.{child_document_table_name}"


# Doc Splitting
# ---------------------------
# Using 8k tokens here, since LLM attention is decreasing quickyly with larger windows.  
n_parent_tokens=1024*8
n_child_tokens= 1024


# SP Token/Host
# ---------------------------
# Get Service Principal Token from secrets service 
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
# Host name for the current workspace
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_HOST"] = host

# Endpoint Names
# ---------------------------
vector_search_endpoint_name = "one-env-shared-endpoint-9"
embedding_endpoint_name = "databricks-bge-large-en"
llm_endpoint_name = "databricks-dbrx-instruct"
feature_serving_endpoint_name = "allianz-features-parent-splits_aj"
serving_endpoint_name = f"allianz-copilot"

emb_dim = 1024

# Vector Store / Child Documents / Similarity Search
# ---------------------------
# Path to the index in UC 
vs_index_name =f"{catalog}.{schema}.docs_index"
# How many document's similarity search will return (you can also specify a confidence threshold)
n_top_results = 3
# Name of the column that contains the text/natural language in vector index
text_column= "text"
text_vector= "embedding"
#-----------------------------


# Full path to the volume
volume_path_raw_data = f"/Volumes/{catalog}/{schema}/{raw_data_volume}"
volume_path_checkpoints = f"/Volumes/{catalog}/{schema}/{checkpoints_volume}"


# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient(host=host, token=os.environ['DATABRICKS_TOKEN'])
sp_id = w.current_user.me().emails[0].value
print(f"Service Principal ID: {sp_id}")

# COMMAND ----------

# Use the user group instead of the SP directly (make sure the SP is part of the user group) 
sp_id = "allianz-test"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Tables

# COMMAND ----------

# Create catalog if not exists
#spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Use catalog 
spark.sql(f"USE CATALOG {catalog}")

# Create schema if not exists within the catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

# Use schema 
spark.sql(f"USE SCHEMA {schema}")

# Create volume if not exists within the catalog
spark.sql(f"CREATE VOLUME IF NOT EXISTS {raw_data_volume}")

# Create volume if not exists within the catalog
spark.sql(f"CREATE VOLUME IF NOT EXISTS {checkpoints_volume}")

# COMMAND ----------

# Enable Service Principal (SP) to use the database, select from table and execute model
spark.sql(f"GRANT USAGE ON CATALOG {catalog} TO `{sp_id}`")
spark.sql(f"GRANT USE SCHEMA ON DATABASE {catalog}.{schema} TO `{sp_id}`")
spark.sql(f"GRANT CREATE TABLE ON DATABASE {catalog}.{schema} TO `{sp_id}`")
spark.sql(f"GRANT EXECUTE ON DATABASE {catalog}.{schema} TO `{sp_id}`")
spark.sql(f"GRANT SELECT ON DATABASE {catalog}.{schema} TO `{sp_id}`")

# COMMAND ----------

# DBTITLE 1,Create Document Table
# Create parent document table 
spark.sql(f"""
CREATE OR REPLACE TABLE {document_table}  (
    document_id BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 0 INCREMENT BY 1) PRIMARY KEY,
    {text_column} STRING NOT NULL,
    source STRING NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');
""")

# Set table properties to do change data capture (required for online table) 
spark.sql(f"""
ALTER TABLE {document_table} SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);
""")

# COMMAND ----------

# DBTITLE 1,Create Parent Document Table
# Create parent document table 
# We use an auto increment PK to generate unqique identifier for each parent_split
spark.sql(f"""
CREATE OR REPLACE TABLE {parent_document_table} (
    parent_split_id BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 0 INCREMENT BY 1) PRIMARY KEY,
    document_id BIGINT NOT NULL CONSTRAINT document_id_fk FOREIGN KEY REFERENCES documents,
    parent_split_index INT NOT NULL, 
    {text_column} STRING NOT NULL,
    source STRING NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');
""")

# Set table properties to do change data capture (required for online table) 
spark.sql(f"""
ALTER TABLE {parent_document_table} SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);
""")

# COMMAND ----------

# DBTITLE 1,Create Child Document Table

# Create parent document table 
spark.sql(f"""
CREATE OR REPLACE TABLE {child_document_table} (
    child_split_id BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 0 INCREMENT BY 1) PRIMARY KEY,
    parent_split_id BIGINT NOT NULL CONSTRAINT parent_split_id_fk FOREIGN KEY REFERENCES {parent_document_table},
    document_id BIGINT NOT NULL CONSTRAINT document_id_child_fk FOREIGN KEY REFERENCES {document_table},
    child_split_index INT NOT NULL,
    parent_split_index INT NOT NULL, 
    {text_vector} ARRAY<DOUBLE> NOT NULL, 
    {text_column} STRING NOT NULL,
    source STRING NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported');
""")

# Set table properties to do change data capture (required for online table) 
spark.sql(f"""
ALTER TABLE {child_document_table} SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);
""")

# COMMAND ----------

# Make SP owner of the tables
#spark.sql(f"ALTER TABLE {parent_document_table} OWNER TO `{sp_id}`")
#spark.sql(f"ALTER TABLE {child_document_table} OWNER TO `{sp_id}`")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 01_Load Data 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Load Documents
# MAGIC
# MAGIC We put AutoLoader on the the path of the raw documents in the volume. This allows us to keep track of the documents that we already have loaded. When we run the code the new documents will be added to the tables and split being created. 

# COMMAND ----------

from pyspark.sql.functions import input_file_name, collect_list

# Configure Auto Loader to ingest multiple .txt files
df_documents = spark.readStream.format("cloudFiles") \
    .option("cloudFiles.format", "text") \
    .option("wholeText", "true") \
    .option("pathGlobFilter", "*.txt") \
    .option("maxFilesPerTrigger", "8") \
    .load(volume_path_raw_data)

# Add the source column with the file name
df_documents_with_source = df_documents.withColumn("source", input_file_name())

# Rename the 'value' column to 'text'
df_documents_with_text = df_documents_with_source.withColumnRenamed("value", "text")

# Select only the necessary columns for further processing
df_documents_final = df_documents_with_text.select("text", "source")

# COMMAND ----------

checkpoint_path = f"/Volumes/{catalog}/{schema}/checkpoints/stream_checkpoint/{document_table_name}_table"

# Configure the writeStream operation
write_query = df_documents_final\
    .writeStream \
    .trigger(availableNow=True) \
    .format("delta") \
    .option("checkpointLocation", checkpoint_path) \
    .outputMode("append") \
    .option("truncate", "false") \
    .toTable(document_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Parent Splits 

# COMMAND ----------

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader, DataFrameLoader
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# COMMAND ----------

# Get document and document_parent_split tables 
df_documents = spark.table(document_table)
df_parent_split = spark.table(parent_document_table)

# Get only the new documents 
new_documents = df_documents.join(df_parent_split, df_parent_split.document_id == df_documents.document_id, "left_outer") \
    .filter(df_parent_split.document_id.isNull()) \
    .select(df_documents["*"])


# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

# Define the Pandas UDF
def split_parent_documents(pdf):
    # Get the document id of the current execution 
    document_id_value = pdf["document_id"].values[0]

    # Load parent document and split it into child documents
    documents = DataFrameLoader(pdf, page_content_column="text").load()

    # Create parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=n_parent_tokens, chunk_overlap=0)
    parent_documents = parent_splitter.split_documents(documents)

    # Create id lists 
    n_docs = len(parent_documents)  
    document_id_list = [document_id_value] * n_docs

    # Get the text and document id's from the the parent document object
    text = []
    source = []
    parent_split_index = []

    # For each parent document append document_id, text chunk and source 
    for i, doc in enumerate(parent_documents):
        text.append(doc.page_content)
        source.append(doc.metadata["source"])
        parent_split_index.append(i)

    # Create pandas df
    df = pd.DataFrame(
        {
            "document_id": document_id_list,
            "parent_split_index": parent_split_index,
            "text": text,
            "source": source,
        }
    )

    return df

# Define output schema
output_schema = "document_id bigint, parent_split_index int, text string, source string"

# COMMAND ----------

from pyspark.sql import functions as F

# Group by parent_split_id and apply the Pandas UDF to split into parent chunks
df_parent_splits = (
    new_documents.groupby("document_id")
    .applyInPandas(split_parent_documents, schema=output_schema)
    .orderBy("document_id", "parent_split_index")
)

# COMMAND ----------

df_parent_splits.write.format("delta").mode("append").saveAsTable(parent_document_table)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Child Splits 

# COMMAND ----------

df_parent_split = spark.table(parent_document_table)
df_child_split = spark.table(child_document_table)


# Get only the new documents 
new_parent_documents = df_parent_split.join(df_child_split, df_child_split.parent_split_id == df_parent_split.parent_split_id, "left_outer") \
    .filter(df_child_split.parent_split_id.isNull()) \
    .select(df_parent_split["*"])

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

# Define the Pandas UDF
def split_child_documents(pdf):
    # Get the document id of the current execution 
    document_id = pdf["document_id"].values[0]
    parent_id_value = pdf["parent_split_id"].values[0]
    parent_split_index = pdf["parent_split_index"].values[0]

    # Load parent document and split it into child documents
    documents = DataFrameLoader(pdf, page_content_column="text").load()

    # Create parent splitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=n_child_tokens, chunk_overlap=0)
    child_documents = child_splitter.split_documents(documents)

    # Create id lists 
    n_docs = len(child_documents)
    
    document_id_list = [document_id] * n_docs
    parent_id_list = [parent_id_value] * n_docs
    parent_split_index_list = [parent_split_index] * n_docs

    # Get the text and document id's from the the parent document object
    child_split_index = []
    text = []
    source = []

    # For each parent document append document_id, text chunk and source 
    for i, doc in enumerate(child_documents):
        child_split_index.append(i)
        text.append(doc.page_content)
        source.append(doc.metadata["source"])

    # Create embeddings from text
    embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)

    embeddings = embedding_model.embed_documents(text)

    # Create pandas df
    df = pd.DataFrame(
        {
            "parent_split_id": parent_id_list,
            "document_id": document_id_list,
            "child_split_index": child_split_index,
            "parent_split_index": parent_split_index_list,
            "embedding": embeddings,
            "text": text,
            "source": source,
        }
    )

    return df

# Define output schema
output_schema = "parent_split_id bigint, document_id bigint, child_split_index int, parent_split_index int, embedding array<double> ,text string, source string"

# COMMAND ----------

from pyspark.sql import functions as F

# Group by parent_split_id and apply the Pandas UDF to split into parent chunks
df_child_splits = (
    new_parent_documents.groupby("parent_split_id")
    .applyInPandas(split_child_documents, schema=output_schema)
    .orderBy("document_id", "parent_split_id", "child_split_index")
)

# COMMAND ----------

df_child_splits.write.format("delta").mode("append").saveAsTable(child_document_table)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 02_Create Online Stores 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Parent Documents Online Store

# COMMAND ----------

# DBTITLE 1,Create Online Table
from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient(host=host, token=os.environ['DATABRICKS_TOKEN'])

# Create an online table
# We use mode triggered: Will write changes to online table if sometheting changes in source table
spec = OnlineTableSpec(
  primary_key_columns=["parent_split_id"],
  source_table_full_name=parent_document_table,
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
)

try:
  w.online_tables.create(name=f"{parent_document_table}_online", spec=spec)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

pprint(w.online_tables.get(f"{parent_document_table}_online"))

# COMMAND ----------

feature_serving_endpoint_name

# COMMAND ----------

# DBTITLE 1,Create Feature Serving Endpoint
# Import necessary classes
from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


# Create a feature store client
fe = FeatureEngineeringClient()

# Create a lookup to fetch features by key
features=[
  FeatureLookup(
    table_name=parent_document_table,
    lookup_key="parent_split_id",
    feature_names=[text_column]
  )
]

# Create feature spec with the lookup for features
document_spec_name = f"{catalog}.{schema}.parent_document_splits_spec"

try:
  fe.create_feature_spec(name=document_spec_name, features=features, exclude_columns=["document_id"])
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# Create endpoint for serving parent documents
try:
  endpoint_config = EndpointCoreConfigInput(
    name=feature_serving_endpoint_name, 
    served_entities=[
       ServedEntityInput(
          entity_name=document_spec_name, 
          workload_size="Small", 
          scale_to_zero_enabled=False,  
          environment_vars={
                 "DATABRICKS_TOKEN": "{{secrets/genai/rag_sp_token}}",
                 "DATABRICKS_HOST": f"{host}"
          }
        )
      ]
    )
  
  status = w.serving_endpoints.create_and_wait(name=feature_serving_endpoint_name, config=endpoint_config)

  # Print endpoint creation status
  print(status)

except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Child Documents Online Store

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Get embedding endpoint 
embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)

# Get VectorSearchClient 
vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])


# COMMAND ----------

# DBTITLE 1,Create Vector Index

def create_get_index(vsc, vector_search_endpoint_name, vs_index_name, primarky_key="child_split_id",text_column="text", text_vector="embedding"):
  # Check if index already exists, if not create index 
  if not index_exists(vsc, vector_search_endpoint_name, vs_index_name):
    print(f"Creating index {vs_index_name} on endpoint {vector_search_endpoint_name}...")

    index = vsc.create_delta_sync_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=vs_index_name,
        primary_key=primarky_key,
        source_table_name=child_document_table,
        pipeline_type='TRIGGERED',
        embedding_dimension=emb_dim,
        embedding_vector_column=text_vector,
        embedding_source_column=text_column,
        embedding_model_endpoint_name=embedding_model,
    )

  # Get index if exists 
  index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_name)

  return index


# COMMAND ----------

index = create_get_index(
    vsc=vsc,
    vector_search_endpoint_name=vector_search_endpoint_name,
    vs_index_name=f"{child_document_table}_index",
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 03_Build Langchain Retriever and Q&A Chain

# COMMAND ----------

# DBTITLE 1,Custom Retriever
import os
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
import mlflow.deployments
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from urllib.parse import quote

class DatabricksParentDocumentRetriever(BaseRetriever):
    """
    A class for retrieving relevant parent documents based on a query using Databricks Vector Search and Online Table Serving.

    Args:
        vs_index (object): The DatabricksVectorSearch index.
        embedding_model (object): The DatabricksEmbeddings model for generating query embeddings.
        deploy_client (object): The MLflow deployment client.
        feature_endpoint_name (str): The name of the feature serving endpoint.
        parent_id_key (str): The key for the parent document ID.

    Attributes:
        vs_index (object): The DatabricksVectorSearch index.
        embedding_model (object): The DatabricksEmbeddings model for generating query embeddings.
        deploy_client (object): The MLflow deployment client.
        feature_endpoint_name (str): The name of the feature serving endpoint.
        parent_id_key (str): The key for the parent document ID.
    """

    vs_index: object
    embedding_model: object
    deploy_client: object
    feature_endpoint_name: str
    parent_id_key: str

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieves relevant parent documents based on a query.

        Args:
            query (str): The query string.
            run_manager (CallbackManagerForRetrieverRun): The run manager for managing callbacks during retrieval.

        Returns:
            List[Document]: A list of relevant parent documents.
        """

        if isinstance(query, dict):
            question = query["question"]
            filename = query["filename"]
        else:
            question = query
            filename = None

        # Create embedding vector from query
        # ----------------------------------
        embedding_vector = self.embedding_model.embed_query(question)

        # Look for similar vectors, return parent IDs
        # ----------------------------------
        filters = {}
        if filename is not None:
            filters = {"source LIKE": quote(filename)}

        resp = self.vs_index.similarity_search(
            columns=[self.parent_id_key],
            query_vector=embedding_vector,
            num_results=n_top_results,
            filters=filters,
        )

        data = resp and resp.get("result", {}).get("data_array")

        # Create unique set of IDs so we are not retrieving the same document twice
        parent_document_ids = list(set([
            int(document_id) for document_id, distance in data
        ]))

        # Get parent context from feature serving endpoint
        # ----------------------------------

        # Put id's into TF format to query the feature serving endpoint
        ids = {
            "dataframe_records": [
                {self.parent_id_key: id} for id in parent_document_ids
            ]
        }

        # Query Feature Serving Endpoint
        parent_content = self.deploy_client.predict(
            endpoint=self.feature_endpoint_name, inputs=ids
        )

        # Convert into langchain docs using list comprehension
        result_docs = [Document(doc["text"]) for doc in parent_content["outputs"]]

        return result_docs

# COMMAND ----------

# DBTITLE 1,Build Retriever Function
from databricks.vector_search.client import VectorSearchClient

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host

    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])

    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=f"{catalog}.{schema}.documents_child_split_index"
    )

    embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)

    # Get mlflow client 
    client = mlflow.deployments.get_deploy_client("databricks")

    retrevier = DatabricksParentDocumentRetriever(vs_index=vs_index, embedding_model=embedding_model, deploy_client=client, feature_endpoint_name=feature_serving_endpoint_name, parent_id_key="parent_split_id",)

    return retrevier

# COMMAND ----------

retriever = get_retriever()

# COMMAND ----------

retriever.get_relevant_documents(query={"question": "Was sind meine Plfichten?", "filename": "Berichterstattungstermine.txt"})

# COMMAND ----------

from langchain.globals import set_debug
set_debug(False)

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint=llm_endpoint_name, max_tokens = 4096)
print(f"Test chat model: {chat_model.invoke('What is Apache Spark')}")

# COMMAND ----------

retriever.get_input_schema()

# COMMAND ----------

retriever.dict()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Simple Q&A Chain
# MAGIC
# MAGIC For simple validation tasks we can use a Q&A chain. 

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

system_prompt = """You are an assistant for Allianz Direct Insurance customers. You are answering insurance related questions to Allianz Direct. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Respond it the language the question was asked. 
Use the following pieces of context to answer the user's question.

`Context`

{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat_model, prompt)

retrieval = RunnableParallel(
    {
      "context": {"question": itemgetter("question"), "filename": itemgetter("filename")} | get_retriever(), 
      "question": RunnablePassthrough()
    }
)

chain = retrieval | question_answer_chain

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda

system_prompt = """You are an assistant for Allianz Direct Insurance customers. You are answering insurance related questions to Allianz Direct. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Respond it the language the question was asked. 
Use the following pieces of context to answer the user's question.

`Context`

{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat_model, prompt)

question_getter = itemgetter("question")
filename_getter = itemgetter("filename")

retrieval = RunnableParallel(
    {
      "context": {"question": question_getter, "filename": filename_getter} | get_retriever(),
      "question": RunnablePassthrough()
    }
)

chain = retrieval | question_answer_chain

# COMMAND ----------

question_answer_chain.

# COMMAND ----------

question = {"question": "Was ist Allianz Direct-Reiseversicherung?", "filename": "Pflichten.txt"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

#from langchain.chains import RetrievalQA
#from langchain.prompts import PromptTemplate
#from langchain_community.chat_models import ChatDatabricks

#TEMPLATE = """You are an assistant for Allianz Direct Insurance customers. You are answering insurance related questions to Allianz Direct. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Respond it the language the question was asked. 
#Use the following pieces of context to answer the question at the end:
#{context}
#
#Question: {question}
#
#Answer:
#"""
#prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

#chain = RetrievalQA.from_chain_type(
#    llm=chat_model,
#    chain_type="stuff",
#    retriever=retriever,
#    chain_type_kwargs={"prompt": prompt}
#)

# COMMAND ----------

# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"question": "Was ist Allianz Direct-Reiseversicherung?", "filename": "Pflichten.txt"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"question": "Bin ich versichert, wenn meine Großmutter während meiner Reise stirbt? Wenn ja, welche Bedingungen gelten?", "filename": "AVB22 - test.txt"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Create MLFlow Model Registry if doesn't exist
import mlflow

def create_model_registry(registry_name):
    client = mlflow.tracking.MlflowClient()

    try:
        model_registry = client.get_registered_model(registry_name)
        print("Model registry already exists.")
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(registry_name)
        print("Model registry created successfully.")

# Usage example
registry_name = "allianz_model_registry_copilot"
create_model_registry(registry_name)

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain
import pydantic

#mlflow.set_registry_uri("databricks-uc")
mlflow.set_registry_uri("databricks://DEFAULT")
#model_name = f"{catalog}.{schema}.chatbot_model"

with mlflow.start_run(run_name=f"{catalog}_{schema}") as run:

    signature = infer_signature(question, answer)

    model_info = mlflow.langchain.log_model(
        lc_model= chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=registry_name,
        #registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic=="+pydantic.__version__,
        ],
        input_example=question,
        signature=signature,
        metadata={"task":"llm/v1/chat"}
    )

# COMMAND ----------

import mlflow

# Usage example
registry_name = "allianz_model_registry_copilot"
model_version = 3

model_uri = f"models:/{registry_name}/{model_version}"
model = mlflow.langchain.load_model(model_uri)


# COMMAND ----------

# DBTITLE 1,Create Conversational or Q&A Endpoint
# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput,ServedModelInputWorkloadSize

latest_model_version=1

endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            #model_name=model_name,
            model_name=registry_name,
            model_version=latest_model_version, 
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=False,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}",
                "DATABRICKS_HOST": f"{host}"
            }
        )
    ]
)
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"

if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conversional Chatbot

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is an Insurance for?"}))

# COMMAND ----------

prompt_with_history_str = """
Your are a Insurance Assistant. Please answer question related to the chat histroy or context only. If you don't know don't answer. Respond with the same language the question was aked.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "What is an Insurance for?"}, 
        {"role": "assistant", "content": "Insurance is a means of protection from financial loss. It is a form of risk management, primarily used to hedge against the risk of a contingent or uncertain loss."}, 
        {"role": "user", "content": "Do Insurances protect me from any form of financial loss?"}
    ]
}))

# COMMAND ----------



is_question_about_allianz_str = """
You are classifying documents to know if this question is related to your work as insurance assistant. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is the Allianz Direct travel cancellation insurance?, classify this question: Which risks will this contract cover?
Expected Response: Yes

Question: Knowing this followup history: What is Allianz Direct?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_allianz_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_allianz_str
)

is_about_allianz_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_allianz_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Allianz Direct: 
print(is_about_allianz_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is an Insurance for?"}, 
        {"role": "assistant", "content": "Insurance is a means of protection from financial loss. It is a form of risk management, primarily used to hedge against the risk of a contingent or uncertain loss."}, 
        {"role": "user", "content": "Do Insurances protect me from any form of financial loss?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Databricks
print(is_about_allianz_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
pprint(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "Was ist ein Assistenzhund?"}]}))

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language and in the language the question was asked. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is an Insurance for?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is an Insurance for?"}, 
        {"role": "assistant", "content": "Insurance is a means of protection from financial loss. It is a form of risk management, primarily used to hedge against the risk of a contingent or uncertain loss."}, 
        {"role": "user", "content": "Do Insurances protect me from any form of financial loss?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for Insurance customers and agents. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])


relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer this question, because it is not related to my insurance assistant tasks.'})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_allianz_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is an Insurance for?"}, 
        {"role": "assistant", "content": "Insurance is a means of protection from financial loss. It is a form of risk management, primarily used to hedge against the risk of a contingent or uncertain loss."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is an Insurance for?"}, 
        {"role": "assistant", "content": "Insurance is a means of protection from financial loss. It is a form of risk management, primarily used to hedge against the risk of a contingent or uncertain loss."}, 
        {"role": "user", "content": "Which risks will Allianz cover?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

conversation = {
    "messages": [
        {"role": "user", "content": "Für was braucht man eine Versicherung"}, 
        {"role": "assistant", "content": "Versicherungen sichern finanzielle Risiken ab..."}, 
        {"role": "user", "content": "Welche Bereiche der Versicherung sind von einem Assitenzhund betroffen?"},
        {"role": "assistant", "content":  """Assistenzhunde sind in der Allianz Direct Reise-Rücktrittsversicherung in verschiedenen Bereichen relevant. Sie sind beispielsweise in den Definitionen von "Familienmitglied" und "Krankenhaus" erwähnt. In der Regel sind Assistenzhunde jedoch nicht direkt von der Versicherung abgedeckt, sondern dienen als Unterstützung für versicherte Personen."""},
        {"role": "user", "content":  "Was ist die Definition des Assitenzhundes im Bezug auf Krankenhaus?"}
    ]
}

print(f'Testing with relevant history and question...')
response = full_chain.invoke(conversation)
display_chat(conversation["messages"], response)

# COMMAND ----------

# DBTITLE 1,Log as Langchain
import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.conversational_model"

with mlflow.start_run(run_name=f"{catalog}_{schema}_rag") as run:
    #Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
        metadata={"task":"llm/v1/chat"}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create/Update Conversational or Q&A Endpoint

# COMMAND ----------

os.environ["DATABRICKS_HOST"]

# COMMAND ----------

model_name

# COMMAND ----------


1. Chatbot --> conditions: 1.a valid strategy for large documents / 1.b security. 
2. Create a question set with corresponding answers and quote of the document that aligns with the RAG data --> judge model / kpi's 
3. Sucess criterias / up discussion 

----------------

--> embedding size --> smaller chunks --> have efficient retrieval --> use parent document/ large window size 

# COMMAND ----------

input = {
  "dataframe_split": {
    "columns": [
      "query"
    ],
    "data": [
      [
        "Was ist ein Assistenzhund"
      ]
    ]
  }
}


