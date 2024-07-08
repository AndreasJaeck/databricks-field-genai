# Databricks notebook source
# MAGIC %md 
# MAGIC ## Evaluate GenAI with LLM-as-a-Judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge.png?raw=true" style="float: right" width="900px">

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch==0.22 langchain==0.1.5 databricks-sdk==0.18.0 mlflow==2.13.0 pandas==1.5 tiktoken==0.5.1 textstat==0.7.3 evaluate==0.4.1 torch==2.0.1 transformers==4.30.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pyspark.sql.functions as F
import mlflow
import os

# COMMAND ----------

judge_name = "openai"  # gpt4

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Models
# MAGIC #### Own RAG models from UC

# COMMAND ----------


os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("alan_maz_secrets", "rag_chatbot")

model_name = "alan_demos.genai.rag_chatbot_model"
llama_rag_6 = mlflow.langchain.load_model(f"models:/{model_name}/6")
llama_rag_7 = mlflow.langchain.load_model(f"models:/{model_name}/7")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Off the shelf DBRX and GPT4

# COMMAND ----------

from langchain.chat_models import ChatDatabricks


dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=500)
judge = ChatDatabricks(endpoint=judge_name, max_tokens=500)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Data for Evaluation
# MAGIC
# MAGIC Contains labels which is optional but allows to calculate metrics such as "answer correctness"

# COMMAND ----------

eval_data = (spark.read.table("alan_demos.genai.evaluation_dataset")
             .selectExpr('question as inputs', 'answer as targets')
             .sample(fraction=0.005, seed=40)
             .toPandas()
)
display(eval_data)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Apply models to data 

# COMMAND ----------

eval_data["llama_rag_7"] = eval_data["inputs"].apply(lambda x: llama_rag_7.invoke(x)['result'])
eval_data["llama_rag_6"] = eval_data["inputs"].apply(lambda x: llama_rag_6.invoke(x)['result'])

eval_data["dbrx"] = eval_data["inputs"].apply(lambda x: dbrx.invoke(x).content)
eval_data["judge"] = eval_data["inputs"].apply(lambda x: judge.invoke(x).content)

# COMMAND ----------

display(eval_data)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create Metrics 
# MAGIC #### Build-in metric

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

professionalism = make_genai_metric(
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

results = []
for model in ["llama_rag_6", "llama_rag_7", "dbrx", "judge"]:

  with mlflow.start_run(run_name=model) as run:
    result = mlflow.evaluate(
            data=eval_data,
            predictions=model,
            targets="targets",
            model_type="question-answering",  # exact-match, toxicity, ari_grade_level, felsch_kincaid_grade_level
            extra_metrics=[answer_correctness_metric, professionalism]
    )

  results.append(result)

# COMMAND ----------

display(results[0].tables["eval_results_table"])

# COMMAND ----------


