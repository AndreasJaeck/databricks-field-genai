# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Load Data
# MAGIC
# MAGIC This notebook will load new .txt files from the Volume, fill the tables and sync the online stores. Usually this notebook will be executed as a job. For example when a new file is added to the Volume or on a fixed schedule. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Install Dependencies 

# COMMAND ----------

# MAGIC %pip install --quiet -U mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get configuration

# COMMAND ----------

import os
import mlflow

#Get the conf from the local conf file
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
secrets_config = model_config.get("secrets_config")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

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

from pyspark.sql.functions import input_file_name, regexp_extract

# Get document table configuration
document_table_schema = retriever_config.get("document_table_schema")

# Configure Auto Loader to ingest multiple .txt files
df_documents = spark.readStream.format("cloudFiles") \
    .option("cloudFiles.format", "text") \
    .option("wholeText", "true") \
    .option("pathGlobFilter", "*.txt") \
    .option("maxFilesPerTrigger", "8") \
    .load(databricks_resources.get("source_volume_path"))

# Add the source uri column with the file name
df_documents_with_source = df_documents.withColumn(document_table_schema.get("document_uri"), input_file_name())

# Extract the document name and append it as the new column 'document_name'
df_documents_with_name = df_documents_with_source.withColumn(
    document_table_schema.get("document_name_col"),
    regexp_extract(input_file_name(), ".*/([^/]+)\\.txt$", 1)
)

# Rename the 'value' column to 'text'
df_documents_with_text = df_documents_with_name.withColumnRenamed("value", document_table_schema.get("text_col"))

# Select only the necessary columns for further processing
df_documents_final = df_documents_with_text.select(
    document_table_schema.get("text_col"), 
    document_table_schema.get("document_uri"), 
    document_table_schema.get("document_name_col")
)

# COMMAND ----------

# Create path for streaming checkpoints 
checkpoint_path = f"{databricks_resources.get('checkpoint_volume_path')}/streaming_checkpoints/"

# Configure the writeStream operation
write_query = df_documents_final\
    .writeStream \
    .trigger(availableNow=True) \
    .format("delta") \
    .option("checkpointLocation", checkpoint_path) \
    .outputMode("append") \
    .option("truncate", "false") \
    .toTable(databricks_resources.get("document_table"))

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

# DBTITLE 1,Only get new records from documents table
# Get document and document_parent_split tables 
df_documents = spark.table(databricks_resources.get("document_table"))
df_parent_split = spark.table(databricks_resources.get("parent_table"))

# Get only the new documents 
new_documents = df_documents.join(df_parent_split, df_parent_split.document_id == df_documents.document_id, "left_outer") \
    .filter(df_parent_split.document_id.isNull()) \
    .select(df_documents["*"])


# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

# Get parent table configuration
parent_table_schema = retriever_config.get("parent_splits_table_schema")
document_foreign_key = parent_table_schema.get("document_foreign_key")
parent_split_index_col = parent_table_schema.get("parent_split_index")
parent_text_col = parent_table_schema.get("text_col")
document_name_col = parent_table_schema.get("document_name_col")
parent_document_uri = parent_table_schema.get("document_uri")


# Define the Pandas UDF
def split_parent_documents(pdf):
    # Get the document id of the current execution 
    document_id_value = pdf[document_table_schema.get("primary_key")].values[0]

    # Load parent document and split it into parent documents
    documents = DataFrameLoader(pdf, page_content_column=document_table_schema.get("text_col")).load()

    # Create parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=retriever_config.get("n_parent_tokens"), chunk_overlap=0)
    parent_documents = parent_splitter.split_documents(documents)

    # Create id lists 
    n_docs = len(parent_documents)  
    document_id_list = [document_id_value] * n_docs

    # Get the text and document id's from the the parent document object
    text = []
    source_name = []
    source = []
    parent_split_index = []

    # For each parent document append document_id, text chunk and source 
    for i, doc in enumerate(parent_documents):
        text.append(doc.page_content)
        source_name.append(doc.metadata[document_name_col])
        source.append(doc.metadata[parent_document_uri])
        parent_split_index.append(i)

    # Create pandas df
    df = pd.DataFrame(
        {
            document_foreign_key: document_id_list,
            parent_split_index_col: parent_split_index,
            parent_text_col: text,
            document_name_col: source_name,
            parent_document_uri: source,
        }
    )

    return df

# Define output schema
output_schema = f"{document_foreign_key} bigint, {parent_split_index_col} int, {parent_text_col} string, {document_name_col} string , {parent_document_uri} string"

# COMMAND ----------

from pyspark.sql import functions as F

# Group by parent_split_id and apply the Pandas UDF to split into parent chunks
df_parent_splits = (
    new_documents.groupby(document_foreign_key)
    .applyInPandas(split_parent_documents, schema=output_schema)
    .orderBy(document_foreign_key, parent_split_index_col)
)

# COMMAND ----------

df_parent_splits.write.format("delta").mode("append").saveAsTable(databricks_resources.get("parent_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Child Splits 

# COMMAND ----------

df_parent_split = spark.table(databricks_resources.get("parent_table"))
df_child_split = spark.table(databricks_resources.get("child_table"))


# Get only the new documents 
new_parent_documents = df_parent_split.join(df_child_split, df_child_split.parent_split_id == df_parent_split.parent_split_id, "left_outer") \
    .filter(df_child_split.parent_split_id.isNull()) \
    .select(df_parent_split["*"])

# COMMAND ----------

# DBTITLE 1,Pandas UDF for Child Splits
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

# Get parent table configuration
child_table_schema = retriever_config.get("child_splits_table_schema")
document_foreign_key = child_table_schema.get("document_foreign_key")
child_split_index_col = child_table_schema.get("child_split_index")
parent_split_foreign_key = child_table_schema.get("parent_split_foreign_key")
parent_split_index_col = child_table_schema.get("parent_split_index")
child_text_col = child_table_schema.get("text_col")
document_name_col = child_table_schema.get("document_name_col")
child_document_uri = child_table_schema.get("document_uri")
embedding_vector_col = child_table_schema.get("embedding_vector_col")


# Define the Pandas UDF
def split_child_documents(pdf):
    # Get the document id of the current execution 
    document_id = pdf[document_table_schema.get("primary_key")].values[0]
    parent_id_value = pdf[parent_table_schema.get("primary_key")].values[0]
    parent_split_index = pdf[parent_table_schema.get("parent_split_index")].values[0]

    # Load parent document and split it into child documents
    documents = DataFrameLoader(pdf, page_content_column=parent_table_schema.get("text_col")).load()

    # Create parent splitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=retriever_config.get("n_child_tokens"), chunk_overlap=0)
    child_documents = child_splitter.split_documents(documents)

    # Create id lists 
    n_docs = len(child_documents)
    
    document_id_list = [document_id] * n_docs
    parent_id_list = [parent_id_value] * n_docs
    parent_split_index_list = [parent_split_index] * n_docs

    # Get the text and document id's from the the parent document object
    child_split_index = []
    text = []
    source_name = []
    source = []

    # For each parent document append document_id, text chunk and source 
    for i, doc in enumerate(child_documents):
        child_split_index.append(i)
        text.append(doc.page_content)
        source_name.append(doc.metadata[document_name_col])
        source.append(doc.metadata[child_document_uri])

    # Create embeddings from text
    embedding_model = DatabricksEmbeddings(endpoint=databricks_resources.get("embedding_endpoint_name"))

    embeddings = embedding_model.embed_documents(text)

    # Create pandas df
    df = pd.DataFrame(
        {
            parent_split_foreign_key: parent_id_list,
            document_foreign_key: document_id_list,
            child_split_index_col: child_split_index,
            parent_split_index_col: parent_split_index_list,
            embedding_vector_col: embeddings,
            child_text_col: text,
            document_name_col: source_name,
            child_document_uri: source,
        }
    )

    return df

# Define output schema
output_schema = f"{parent_split_foreign_key} bigint, {document_foreign_key} bigint, {child_split_index_col} int, {parent_split_index_col} int, {embedding_vector_col} array<double> , {child_text_col} string, {document_name_col} string, {child_document_uri} string"

# COMMAND ----------

# DBTITLE 1,Run Pandas UDF
from pyspark.sql import functions as F

# Group by parent_split_id and apply the Pandas UDF to split into parent chunks
df_child_splits = (
    new_parent_documents.groupby(parent_split_foreign_key)
    .applyInPandas(split_child_documents, schema=output_schema)
    .orderBy(document_foreign_key, parent_split_foreign_key, parent_split_index_col)
)

# COMMAND ----------

# DBTITLE 1,Write child splits to table
df_child_splits.write.format("delta").mode("append").saveAsTable(databricks_resources.get("child_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 02_Sync Online Stores 
# MAGIC
# MAGIC The following code is only required if the created job doesn't contain additional tasks to update online table and vector index

# COMMAND ----------

# DBTITLE 1,Class for running and checking pipeline status
#class PipelineUpdater:
#    def __init__(self, workspace_client):
#        self.workspace_client = workspace_client
#
#    def get_current_state(self, pipeline_id):
#        """Retrieve the current state of the pipeline."""
#        return self.workspace_client.pipelines.get(pipeline_id).latest_updates[0].state.value
#
#    def wait_for_final_state(self, pipeline_id):
#        """Wait for the pipeline to reach a final state ('COMPLETED' or 'FAILED')."""
#        current_state = self.get_current_state(pipeline_id)
#        while current_state not in ['COMPLETED', 'CANCELED', 'FAILED']:
#            print(f"Current status: {current_state}. Waiting for completion...")
#            time.sleep(10)  # Wait for 10 seconds before checking the status again
#            current_state = self.get_current_state(pipeline_id)
#        return current_state
#
#    def start_pipeline_update(self, pipeline_id):
#        """Start the pipeline update and wait for it to reach a final state."""
#        print("Starting pipeline update...")
#        up = self.workspace_client.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=True)
#        return self.wait_for_final_state(pipeline_id)
#
#    def update_pipeline_if_needed(self, pipeline_id):
#        """Check the current state and update the pipeline if it's in 'COMPLETED', 'CANCELED' or 'FAILED' state."""
#        current_state = self.get_current_state(pipeline_id)
#        if current_state in ['COMPLETED','CANCELED','FAILED']:
#            final_state = self.start_pipeline_update(pipeline_id)
#        else:
#            print("Pipeline is currently running. Waiting for it to complete...")
#            final_state = self.wait_for_final_state(pipeline_id)
#
#        # Check the final status
#        if final_state == 'COMPLETED':
#            print("Online table update completed successfully.")
#        elif final_state == 'FAILED':
#            print("Online table update failed.")
#            raise Exception("Pipeline update failed.")
#

# COMMAND ----------

# DBTITLE 1,Get workspace client
#from databricks.sdk import WorkspaceClient
#import time
#import os
#
## Get Service Principal Token from secrets service 
#os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secrets_config.get("secret_scope"), secrets_config.get("secret_key"))
## Get the host from the configuration
#os.environ["DATABRICKS_HOST"] = databricks_resources.get("host")
#
## Get workspace client
#w = WorkspaceClient(host=os.environ['DATABRICKS_HOST'], token=os.environ['DATABRICKS_TOKEN'])
#
## Create an instance of PipelineUpdater
#pipeline_updater = PipelineUpdater(w)

# COMMAND ----------

# DBTITLE 1,Run concurrent sync for parent and child online stores
#from concurrent.futures import ThreadPoolExecutor
#
#def update_pipeline_online_table():
#    pipeline_id = w.online_tables.get(f"{databricks_resources.get('parent_table')}_online").spec.pipeline_id
#    pipeline_updater.update_pipeline_if_needed(pipeline_id)
#
#def update_pipeline_vector_search_index():
#    pipeline_id = w.vector_search_indexes.get_index(databricks_resources.get("vector_search_index")).delta_sync_index_spec.pipeline_id
#    pipeline_updater.update_pipeline_if_needed(pipeline_id)
#
#with ThreadPoolExecutor(max_workers=2) as executor:
#    executor.submit(update_pipeline_online_table)
#    executor.submit(update_pipeline_vector_search_index)
