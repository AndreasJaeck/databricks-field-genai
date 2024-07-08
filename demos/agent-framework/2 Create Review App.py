# Databricks notebook source
# MAGIC %md # Create Review App
# MAGIC
# MAGIC Review App allows you to evaluate your chain with external expert users. We provide a simple UI that can be accessed by SSO users without requiering access to Databricks Workspace. 

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create rag chain config

# COMMAND ----------

import mlflow
import yaml

# Set name for the configuration
config_name= "rag_chain_config"

# Define configuraion
rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": "databricks-meta-llama-3-70b-instruct",
        "vector_search_endpoint_name": "one-env-shared-endpoint-2",
        "catalog": "alan_demos",
        "schema": "genai",
        "model_name": "chain_review_app"
    },
    "input_example": {
        "messages": [{"content": "Databricks Lakehouse Platform", "role": "user"}]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        "llm_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "question"],
    },
    "retriever_config": {
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 5, "query_type": "ann"},
        "schema": {"chunk_text": "content", "document_uri": "url", "primary_key": "id"},
        "vector_search_index": "alan_demos.genai.pdf_vs_index",
    },
}
try:
    with open(f'configs/{config_name}.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Build chain with Review App functionality

# COMMAND ----------

# MAGIC
# MAGIC %%writefile configs/chain.py 
# MAGIC
# MAGIC import os
# MAGIC import mlflow
# MAGIC from operator import itemgetter
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import PromptTemplate
# MAGIC from langchain_core.runnables import RunnablePassthrough
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC #Get the conf from the local conf file
# MAGIC model_config = mlflow.models.ModelConfig(development_config='configs/rag_chain_config.yaml')
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC # Connect to the Vector Search Index
# MAGIC vs_client = VectorSearchClient(disable_notice=True)
# MAGIC vs_index = vs_client.get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=retriever_config.get("vector_search_index"),
# MAGIC )
# MAGIC vector_search_schema = retriever_config.get("schema")
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     columns=[
# MAGIC         vector_search_schema.get("primary_key"),
# MAGIC         vector_search_schema.get("chunk_text"),
# MAGIC         vector_search_schema.get("document_uri"),
# MAGIC     ],
# MAGIC ).as_retriever(search_kwargs=retriever_config.get("parameters"))
# MAGIC
# MAGIC # Required to:
# MAGIC # 1. Enable the RAG Studio Review App to properly display retrieved chunks
# MAGIC # 2. Enable evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=vector_search_schema.get("primary_key"),
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     doc_uri=vector_search_schema.get("document_uri")
# MAGIC )
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = PromptTemplate(
# MAGIC     template=llm_config.get("llm_prompt_template"),
# MAGIC     input_variables=llm_config.get("llm_prompt_template_variables"),
# MAGIC )
# MAGIC
# MAGIC # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_query_string)
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Log chain with MlFlow 
# MAGIC
# MAGIC If we want to use the Review App we have to log the chain with mlflow again. 

# COMMAND ----------

# Create UC path for the model
registered_model_name = f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('model_name')}"

# Log the model to MLflow
with mlflow.start_run(run_name=f"2 Create Review App") as r:
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), "configs/chain.py"
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config="configs/rag_chain_config.yaml",  # Chain configuration
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get(
            "input_example"
        ),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Load Model to evaluate chain

# COMMAND ----------

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's deploy our RAG application and open it for external expert users
# MAGIC

# COMMAND ----------

from databricks import agents

instructions_to_reviewer = f"""### Instructions for Testing the our Databricks Documentation Chatbot assistant

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing our assistant. Your contributions are essential to delivering a high-quality product to our end users."""


# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=registered_model_name, model_version=2, scale_to_zero=True)

# Add the user-facing instructions to the Review App
agents.set_review_instructions(registered_model_name, instructions_to_reviewer)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant stakeholders access to the Mosaic AI Agent Evaluation App
# MAGIC
# MAGIC Now, grant your stakeholders permissions to use the Review App. To simplify access, stakeholders do not require to have Databricks accounts.
# MAGIC

# COMMAND ----------

user_list = ["andreas.jack@databricks.com", "alan.mazankiewicz@databricks.com"]
# Set the permissions.
agents.set_permissions(model_name=registered_model_name, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find review app name
# MAGIC
# MAGIC If you lose this notebook's state and need to find the URL to your Review App, you can list the chatbot deployed:
# MAGIC

# COMMAND ----------

active_deployments = agents.list_deployments()
active_deployment = next((item for item in active_deployments if item.model_name == registered_model_name), None)
if active_deployment:
  print(f"Review App URL: {active_deployment.review_app_url}")

# COMMAND ----------

