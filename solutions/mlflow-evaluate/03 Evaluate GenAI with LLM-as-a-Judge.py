# Databricks notebook source
# MAGIC %md 
# MAGIC ## Evaluate GenAI with LLM-as-a-Judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge.png?raw=true" style="float: right" width="900px">

# COMMAND ----------

# MAGIC %pip install textstat==0.7.3 databricks-genai==1.0.2 openai==1.30.1 langchain==0.2.0 langchain-community==0.2.0 langchain_text_splitters==0.2.0 markdown==3.6
# MAGIC %pip install databricks-sdk==0.27.1
# MAGIC %pip install "transformers==4.37.1" "mlflow==2.12.2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pyspark.sql.functions as F
import mlflow
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evaluation with MLFlow
# MAGIC
# MAGIC This example shows how you can use LangChain in conjunction with MLflow evaluate with custom System Prompt.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Establish Baseline
# MAGIC
# MAGIC We will first establish baseline performance using the standard LLM. If the LLM you fine-tune is available as Foundation Model, you can use the API provided by Databricks directly.
# MAGIC
# MAGIC Because we fine-tuned on mistral or llama2-7B, we will deploy a Serving endpoint using this model, making sure it scales to zero to avoid costs.
# MAGIC
# MAGIC **Attention!** If you use an external model like gpt you don't need to execute the next cell. You can use the name of the exiting baseline model endpoint instead. 

# COMMAND ----------

# Get name of the Endpoint with baseline model 
baseline_model_endpoint = "azure-openai-gpt4"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Get model for evaluation
# MAGIC Because we fine-tuned on mistral or llama2-7B, we will deploy a Serving endpoint using this model, making sure it scales to zero to avoid costs.
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

catalog = "dbdemos_aj"
schema = "dbdemos_llm_fine_tuning"
model_name = "mistralai_mistral_7b_instruct_v0_2"

registered_model_name = f"{catalog}.{schema}.{model_name}"
tuned_endpoint_name = "dbdemos_aj_llm_fine_tuned"

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=tuned_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=1,
            min_provisioned_throughput=0, # The minimum tokens per second that the endpoint can scale down to.
            max_provisioned_throughput=600,# The maximum tokens per second that the endpoint can scale up to.
            scale_to_zero_enabled=True
        )
    ],
    auto_capture_config = AutoCaptureConfigInput(catalog_name=catalog, schema_name=schema, enabled=True, table_name_prefix="fine_tuned_llm_inference" )
)

force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == tuned_endpoint_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {tuned_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=tuned_endpoint_name, config=endpoint_config)
else:
  print(f"endpoint {tuned_endpoint_name} already exist...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=tuned_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) Get the judge model
# MAGIC Finally we need a model that will evaluate the answers of the baseline model and the tuned model. Here we use DBRX but any powerful LLM can be used in this role e.g GPT-4 Turbo

# COMMAND ----------

# Get name of the Endpoint with the judge model
judge_name = "databricks-dbrx-instruct"

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 4) Load Data for Evaluation
# MAGIC
# MAGIC Contains labels which is optional but allows to calculate metrics such as "answer correctness"

# COMMAND ----------

eval_dataset = spark.table("dbdemos_aj.dbdemos_llm_fine_tuning.chat_completion_evaluation_dataset").withColumnRenamed("content", "context").toPandas()
display(eval_dataset)

# COMMAND ----------

# Get 10% of the dataset
eval_dataset_test = eval_dataset.sample(frac=0.01)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4) Generate Answers

# COMMAND ----------

#base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
base_model_name = "gpt4"
# ----------------------------------------------------------------------------------------------------------------------------------------- #
# -- Basic chain for the demo. This should instead be your full, real RAG chain (you want to evaluate your LLM on your final chain) ------- #
# ----------------------------------------------------------------------------------------------------------------------------------------- #
system_prompt = """You are a highly knowledgeable and professional Databricks Support Agent. Your goal is to assist users with their questions and issues related to Databricks. Answer questions as precisely and accurately as possible, providing clear and concise information. If you do not know the answer, respond with "I don't know." Be polite and professional in your responses. Provide accurate and detailed information related to Databricks. If the question is unclear, ask for clarification.\n"""

user_input = "Here is a documentation page that could be relevant: {context}. Based on this, answer the following question: {question}"

def build_chain(llm):
    #mistral doesn't support the system role
    if "mistral" in base_model_name:
        messages = [("user", f"{system_prompt} \n{user_input}")]
    else:
        messages = [("system", system_prompt),
                    ("user", user_input)]
        
    return ChatPromptTemplate.from_messages(messages) | llm | StrOutputParser()
  

def 
  

#Build the chain. This should be your actual RAG chain, querying your index
llm = ChatDatabricks(endpoint=llm_endoint_name, temperature=0.1)
chain = build_chain(llm)

#For each entry, call the endpoint
eval_dataset["prediction"] = chain.with_retry(stop_after_attempt=4) \
                                  .batch(eval_dataset[["context", "question"]].to_dict(orient="records"), config={"max_concurrency": 4})

# COMMAND ----------

#For each entry, call the endpoint
eval_dataset["prediction"] = chain.with_retry(stop_after_attempt=4) \
                                  .batch(eval_dataset[["context", "question"]].to_dict(orient="records"), config={"max_concurrency": 4})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5) Build evaluation metrics

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Build in metric

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness
answer_correctness_metric = answer_correctness(model=f"endpoints:/{judge_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Custom metric

# COMMAND ----------

from mlflow.metrics.genai import make_genai_metric, EvaluationExample

professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism_metric = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{judge_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 6) Build evaluation chain
# MAGIC It is important to evaluate the LLM response in combination with the prompt chain.

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
import pandas as pd
import mlflow

#base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
base_model_name = "gpt4"
# ----------------------------------------------------------------------------------------------------------------------------------------- #
# -- Basic chain for the demo. This should instead be your full, real RAG chain (you want to evaluate your LLM on your final chain) ------- #
# ----------------------------------------------------------------------------------------------------------------------------------------- #
system_prompt = """You are a highly knowledgeable and professional Databricks Support Agent. Your goal is to assist users with their questions and issues related to Databricks. Answer questions as precisely and accurately as possible, providing clear and concise information. If you do not know the answer, respond with "I don't know." Be polite and professional in your responses. Provide accurate and detailed information related to Databricks. If the question is unclear, ask for clarification.\n"""

user_input = "Here is a documentation page that could be relevant: {context}. Based on this, answer the following question: {question}"

def build_chain(llm):
    #mistral doesn't support the system role
    if "mistral" in base_model_name:
        messages = [("user", f"{system_prompt} \n{user_input}")]
    else:
        messages = [("system", system_prompt),
                    ("user", user_input)]
    return ChatPromptTemplate.from_messages(messages) | llm | StrOutputParser()
# --------------------------------------------------------------------------------------------------- #

def eval_llm(llm_endoint_name, llm_judge, eval_dataset):

    #Build the chain. This should be your actual RAG chain, querying your index
    llm = ChatDatabricks(endpoint=llm_endoint_name, temperature=0.1)
    chain = build_chain(llm)

    #For each entry, call the endpoint
    eval_dataset["prediction"] = chain.with_retry(stop_after_attempt=4) \
                                      .batch(eval_dataset[["context", "question"]].to_dict(orient="records"), config={"max_concurrency": 4})


    #starts an mlflow run to evaluate the model
    with mlflow.start_run(run_name="eval_"+llm_endoint_name) as run:
        eval_df = eval_dataset[["question","answer","prediction"]].rename(columns={"question": "inputs"})
        results = mlflow.evaluate(
            data=eval_df,
            targets="answer",
            predictions="prediction",
            extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
                mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
            ],
            evaluators="default"
        )
        return results

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### 6) Run evaluation

# COMMAND ----------

base_model_name

# COMMAND ----------

# Evaluate the base foundation model
baseline_results = eval_llm(
    llm_endoint_name= baseline_model_endpoint,
    eval_dataset= eval_dataset_test,
    llm_judge=judge_name
)

# COMMAND ----------

baseline_results.metrics

# COMMAND ----------

tuned_endpoint_name

# COMMAND ----------

# Evaluate the fine tuned model
fine_tuned_results = eval_llm(
    llm_endoint_name= tuned_endpoint_name,
    eval_dataset= eval_dataset_test,
    llm_judge=judge_name
)

# COMMAND ----------


