# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Build Parent Document Retriever

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Install Dependencies 

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get configuration

# COMMAND ----------

import os
import mlflow

#Get the conf from the local conf file
model_config = mlflow.models.ModelConfig(development_config='config/rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
secrets_config = model_config.get("secrets_config")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

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

# MAGIC %md
# MAGIC
# MAGIC ## Create Feature Serving Endpoint

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


