# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create Tables

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Install Dependencies 

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

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


def create_get_index(vsc, vector_search_endpoint_name, embedding_model, vs_index_name, child_document_table, emb_dim, primarky_key,text_column, text_vector):

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

# MAGIC %md
# MAGIC
# MAGIC ## Set variables and permissions, create db's

# COMMAND ----------

# DBTITLE 1,Get configuration
import os
import mlflow

#Get the conf from the local conf file
model_config = mlflow.models.ModelConfig(development_config='config/rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
secrets_config = model_config.get("secrets_config")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# COMMAND ----------

# DBTITLE 1,Get Service Principal
from databricks.sdk import WorkspaceClient

# Get Service Principal Token from secrets service 
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secrets_config.get("secret_scope"), secrets_config.get("secret_key"))
# Get the host from the configuration
os.environ["DATABRICKS_HOST"] = databricks_resources.get("host")


w = WorkspaceClient(host=databricks_resources.get("host"), token=os.environ['DATABRICKS_TOKEN'])
sp_id = w.current_user.me().emails[0].value
print(f"Service Principal ID: {sp_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Tables

# COMMAND ----------

# Create catalog if not exists
spark.sql(f"CREATE CATALOG IF NOT EXISTS `{databricks_resources.get('catalog')}`")

# Use catalog 
spark.sql(f"USE CATALOG `{databricks_resources.get('catalog')}`")

# Create schema if not exists within the catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{databricks_resources.get('schema')}`")

# Use schema 
spark.sql(f"USE SCHEMA `{databricks_resources.get('schema')}`")

# Create volume if not exists within the catalog
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{databricks_resources.get('source_volume')}`")

# Create volume if not exists within the catalog
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{databricks_resources.get('checkpoint_volume')}`")


# COMMAND ----------

# ToDo: Add the text directory create
folder_path = f"{databricks_resources.get('source_volume')}/text"
dbutils.fs.mkdirs(folder_path)


# COMMAND ----------

# DBTITLE 1,Set Permissions for Service Principal
# Enable Service Principal (SP) to use the database, select from table and execute model
spark.sql(f"GRANT USAGE ON CATALOG {databricks_resources.get('catalog')} TO `{sp_id}`")
spark.sql(f"GRANT USE SCHEMA ON DATABASE {databricks_resources.get('catalog')}.{databricks_resources.get('schema')} TO `{sp_id}`")
spark.sql(f"GRANT CREATE TABLE ON DATABASE {databricks_resources.get('catalog')}.{databricks_resources.get('schema')} TO `{sp_id}`")
spark.sql(f"GRANT EXECUTE ON DATABASE {databricks_resources.get('catalog')}.{databricks_resources.get('schema')} TO `{sp_id}`")
spark.sql(f"GRANT SELECT ON DATABASE {databricks_resources.get('catalog')}.{databricks_resources.get('schema')} TO `{sp_id}`")

# COMMAND ----------

# DBTITLE 1,Create Document Table
# Get document table configuration
document_table_schema = retriever_config.get('document_table_schema')

# Create document table 
spark.sql(f"""
CREATE OR REPLACE TABLE {databricks_resources.get("document_table")}  (
    {document_table_schema.get('primary_key')} BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 0 INCREMENT BY 1) PRIMARY KEY,
    {document_table_schema.get('text_col')} STRING NOT NULL,
    {document_table_schema.get('document_name_col')} STRING NOT NULL,
    {document_table_schema.get('document_uri')} STRING NOT NULL,
    {document_table_schema.get('timestamp_col')} TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported', 'delta.enableChangeDataFeed' = 'true');
""")


# COMMAND ----------

# DBTITLE 1,Create Parent Document Table

# Get parent table configuration
parent_splits_table_schema = retriever_config.get('parent_splits_table_schema')

# Create parent document table 
# We use an auto increment PK to generate unqique identifier for each parent_split
spark.sql(f"""
CREATE OR REPLACE TABLE {databricks_resources.get("parent_table")} (
    {parent_splits_table_schema.get('primary_key')} BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 0 INCREMENT BY 1) PRIMARY KEY,
    {parent_splits_table_schema.get('document_foreign_key')}  BIGINT NOT NULL CONSTRAINT document_id_fk FOREIGN KEY REFERENCES {databricks_resources.get("document_table")},
    {parent_splits_table_schema.get('parent_split_index')} INT NOT NULL, 
    {parent_splits_table_schema.get('text_col')} STRING NOT NULL,
    {parent_splits_table_schema.get('document_name_col')} STRING NOT NULL,
    {parent_splits_table_schema.get('document_uri')} STRING NOT NULL,
    {parent_splits_table_schema.get('timestamp_col')} TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported','delta.enableChangeDataFeed' = 'true');
""")


# COMMAND ----------

# DBTITLE 1,Create Child Document Table

# Get child table configuration
child_splits_table_schema = retriever_config.get('child_splits_table_schema')

# Create child document table 
spark.sql(f"""
CREATE OR REPLACE TABLE {databricks_resources.get("child_table")} (
    {child_splits_table_schema.get('primary_key')}  BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 0 INCREMENT BY 1) PRIMARY KEY,
    {child_splits_table_schema.get('parent_split_foreign_key')} BIGINT NOT NULL CONSTRAINT parent_split_id_fk FOREIGN KEY REFERENCES {databricks_resources.get("parent_table")},
    {child_splits_table_schema.get('document_foreign_key')} BIGINT NOT NULL CONSTRAINT document_id_child_fk FOREIGN KEY REFERENCES {databricks_resources.get("document_table")},
    {child_splits_table_schema.get('child_split_index')} INT NOT NULL,
    {child_splits_table_schema.get('parent_split_index')} INT NOT NULL, 
    {child_splits_table_schema.get('embedding_vector_col')}  ARRAY<DOUBLE> NOT NULL, 
    {child_splits_table_schema.get('text_col')}  STRING NOT NULL,
    {child_splits_table_schema.get('document_name_col')} STRING NOT NULL,
    {child_splits_table_schema.get('document_uri')}  STRING NOT NULL,
    {child_splits_table_schema.get('timestamp_col')} TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported', 'delta.enableChangeDataFeed' = 'true');
""")


# COMMAND ----------

# Make SP owner of the tables
#spark.sql(f"ALTER TABLE {parent_document_table} OWNER TO `{sp_id}`")
#spark.sql(f"ALTER TABLE {child_document_table} OWNER TO `{sp_id}`")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Online Stores 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Parent Documents Online Store

# COMMAND ----------

# DBTITLE 1,Create Online Table
from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient(host=databricks_resources.get("host"), token=os.environ['DATABRICKS_TOKEN'])

# Create an online table
# We use mode triggered: Will write changes to online table if sometheting changes in source table
spec = OnlineTableSpec(
  primary_key_columns=[parent_splits_table_schema.get('primary_key')],
  source_table_full_name=databricks_resources.get("parent_table"),
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),

)

# Create online table name 
online_table_name = f"{databricks_resources.get('parent_table')}_online"

try:
  w.online_tables.create(name=online_table_name, spec=spec)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

pprint(w.online_tables.get(online_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Child Documents Online Store

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.embeddings import DatabricksEmbeddings

# Get embedding endpoint 
embedding_model = DatabricksEmbeddings(endpoint=databricks_resources.get("embedding_endpoint_name"))

# Get VectorSearchClient 
vsc = VectorSearchClient(workspace_url=databricks_resources.get("host"), personal_access_token=os.environ["DATABRICKS_TOKEN"])

# Get child table configuration
child_splits_table_schema = retriever_config.get('child_splits_table_schema')

index = create_get_index(
    vsc=vsc,
    vector_search_endpoint_name=databricks_resources.get("vector_search_endpoint_name"), 
    embedding_model=embedding_model, 
    vs_index_name=databricks_resources.get("vector_search_index"), 
    child_document_table=databricks_resources.get("child_table"), 
    emb_dim=retriever_config.get("embedding_dimension"), 
    primarky_key=child_splits_table_schema.get("primary_key"),
    text_column=child_splits_table_schema.get("text_col"), 
    text_vector=child_splits_table_schema.get("embedding_vector_col")
)
