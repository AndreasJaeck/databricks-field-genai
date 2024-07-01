# Databricks notebook source
# MAGIC %md
# MAGIC # Download data from bucket/git repo to speedup demo setup
# MAGIC This notebook generate a fake dataset for fine tuning on Databricks Documentation using DBRX to generate a question and answer for each documentation page.
# MAGIC
# MAGIC We pre-ran this notebook for you and saved the results in our dataset bucket. We'll download it directly to accelerate the demo dataset preparation.

# COMMAND ----------

# MAGIC %run ./00-setup

# COMMAND ----------

# MAGIC %fs ls /FileStore/quentin

# COMMAND ----------

spark.table('raw_documentation').repartition(1).write.format('parquet').mode('overwrite').save('/FileStore/quentin/doc/raw_documentation')
spark.table('databricks_documentation').repartition(1).write.format('parquet').mode('overwrite').save('/FileStore/quentin/doc/databricks_documentation')
spark.table('training_dataset_question').repartition(1).write.format('parquet').mode('overwrite').save('/FileStore/quentin/doc/training_dataset_question')
spark.table('training_dataset_answer').repartition(1).write.format('parquet').mode('overwrite').save('/FileStore/quentin/doc/training_dataset_answer')

# COMMAND ----------

folder = f"/Volumes/{catalog}/{db}/{volume_name}"

tables_exist = spark.catalog.tableExists("databricks_documentation") and spark.catalog.tableExists("training_dataset_answer") and spark.catalog.tableExists("training_dataset_question")
if not tables_exist:
  

# COMMAND ----------

data_downloaded = False
if not data_exists:
    try:
        DBDemos.download_file_from_git(folder+'/raw_documentation', "databricks-demos", "dbdemos-dataset", "/llm/databricks-documentation")
        data_downloaded = True
    except Exception as e: 
        print(f"Error trying to download the file from the repo: {str(e)}. Will generate the data instead...")    

# COMMAND ----------

# MAGIC %md 
# MAGIC #1 Data collection and preparation
# MAGIC
# MAGIC In this notebook, we'll download Databricks Documentation and use this as our Dataset.
# MAGIC
# MAGIC To generate our Training & Testing Dataset containing question and answer for each documentation section, we'll be using Databricks `ai_query` and ask the DBRX Instruct LLM to ask a question and answer it (make sure you check the LLM license used by your SQL AI function).
# MAGIC
# MAGIC *Note: For model fine-tuning, this is typically something you'd instead do manually, using real/human questions and answers. We will instead generate them to simplify our demo.*

# COMMAND ----------

# MAGIC %pip install beautifulsoup4 tiktoken==0.7.0 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Extracting databricks documentation sitemap & pages
# MAGIC
# MAGIC Let's parse docs.databricks.com website and download the html content.

# COMMAND ----------

import requests
import xml.etree.ElementTree as ET

# Fetch the XML content from sitemap
response = requests.get("https://docs.databricks.com/en/doc-sitemap.xml")
root = ET.fromstring(response.content)
# Find all 'loc' elements (URLs) in the XML
urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
print(f"{len(urls)} Databricks documentation pages found")

#Let's split to 200 documentation page to make this demo faster:
#urls = urls[:200]

# COMMAND ----------

# DBTITLE 1,Download Databricks Documentation HTML pages
import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
import re

# Function to fetch HTML content for a given URL
def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to process a URL and extract text from the specified div
def process_url(url):
    html_content = fetch_html(url)
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        article_div = soup.find("div", itemprop="articleBody")
        if article_div:
            article_text = str(article_div)
            return {"url": url, "text": article_text.strip()}
    return None

# Use a ThreadPoolExecutor with 10 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(executor.map(process_url, urls))

# Filter out None values (URLs that couldn't be fetched or didn't have the specified div)
valid_results = [result for result in results if result is not None]

#Save the content in a raw table
# Assuming 'valid_results' is a DataFrame that you want to save
spark.createDataFrame(valid_results) \
    .write \
    .mode('overwrite') \
    .saveAsTable('raw_documentation')

# Display the first 2 rows of the overwritten table
display(spark.table('raw_documentation').limit(2))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * FROM raw_documentation

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation  (id BIGINT GENERATED BY DEFAULT AS IDENTITY, url STRING, content STRING, title STRING);

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Splitting documentation pages in small chunks
# MAGIC LLM models typically have a maximum window size and you won't be able to compute embbeddings for too long texts.
# MAGIC
# MAGIC In addition, the bigger your context is, the longer your inference will be.
# MAGIC
# MAGIC Data preparation is key for your model to perform well and multiple strategy exist depending of your dataset:
# MAGIC
# MAGIC - Split document in small chunks (paragraph, h2...)
# MAGIC - Truncate documents to a fix number
# MAGIC - It could sometime make sense to split in big chunks ans ask a model to summurize each chunks as a one-off job for faster live inference.
# MAGIC
# MAGIC The number of token depends of your model. LLMs are shipped with a Tokenizer that you can use to count how many tokens will be created for a given sequence (usually > number of words) (see [hugging face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [open AI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC
# MAGIC Make sure the tokenizer and context size limit you'll be using here matches your embedding model. Let's try an exemple with GPT3.5 tokenizer: `tiktoken.encoding_for_model("gpt-3.5-turbo")`

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Splitting our big documentation page in smaller chunks (h2 sections)
# MAGIC
# MAGIC In this demo, we have a few big documentation article, too big for our models. We'll split these articles between HTML h2 chunks, and ensure that each chunk isn't bigger than 4000 tokens.<br/>
# MAGIC To do so, we'll be using the `tiktoken` librairy to count gtp3.5 tokens. 
# MAGIC
# MAGIC Let's also remove the HTML tags to send plain text to our model.

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 1500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2, but merge small h2 chunks together to avoid too small. 
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=1500):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # Merge chunks together to add text before h2 and avoid too small docs.
  for c in h2_chunks:
    # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size*0.8:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # Discard too small chunks
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]
 
# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's now split our entire dataset using this functio and a pandas UDF.
# MAGIC
# MAGIC We will also extract the title from the page (based on h1)

# COMMAND ----------

import pyspark.sql.functions as F

# Let's create a user-defined function (UDF) to chunk all our documents with spark
@F.pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    
(spark.table("raw_documentation")
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable("databricks_documentation"))

display(spark.table("databricks_documentation"))

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM databricks_documentation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's use Databricks AI_GENERATE_TEXT to generate Questions for each documentation page  

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS training_dataset_question (id BIGINT GENERATED ALWAYS AS IDENTITY, doc_id BIGINT, question STRING);
# MAGIC CREATE TABLE IF NOT EXISTS training_dataset_answer   (id BIGINT GENERATED ALWAYS AS IDENTITY, question_id BIGINT, answer STRING);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION GENERATE_QUESTION_FROM_DOC(doc STRING)
# MAGIC RETURNS STRING
# MAGIC RETURN
# MAGIC     ai_query(
# MAGIC         "databricks-dbrx-instruct",
# MAGIC         CONCAT(
# MAGIC             "Here is a documentation page from Databricks: ", doc,
# MAGIC             "\nWrite one questions a Databricks user (developer, data engineer, SQL Analyst, data scientist or devops) might ask themselves ",
# MAGIC             "which can be answered from the above documentation.",
# MAGIC             "\nGenerate one questions that can be answered as a user would ask. Just generate the question, no additional text."
# MAGIC         )
# MAGIC     )

# COMMAND ----------

import pyspark.sql.functions as F

spark.table('databricks_documentation').repartition(100) \
     .withColumn("question", F.expr("GENERATE_QUESTION_FROM_DOC(content)")) \
     .selectExpr("id as doc_id", "question") \
     .write.mode('overwrite').saveAsTable('training_dataset_question')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from training_dataset_question

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the Answers based on the doc & the questions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION GENERATE_ANSWER_FROM_DOC(doc STRING, question STRING)
# MAGIC RETURNS STRING
# MAGIC RETURN 
# MAGIC     ai_query(
# MAGIC         "databricks-dbrx-instruct",
# MAGIC         CONCAT(
# MAGIC             "You are an assistant for python, spark, data engineering, setup/install and data science on Databricks. Here is a documentation page with potential relevant informations: \n\n ", doc,
# MAGIC             "\n\nUsing this information, answer the following question: \n\n",
# MAGIC             question
# MAGIC         )
# MAGIC )

# COMMAND ----------

from pyspark.sql import functions as F

df_q = spark.table('training_dataset_question').alias('q')
df_d = spark.table('databricks_documentation').alias('d')
question_df = df_q.join(df_d, df_q['q.doc_id'] == df_d['d.id'])

(question_df.repartition(100)
            .withColumn("answer", F.expr("GENERATE_ANSWER_FROM_DOC(content, question)"))
            .selectExpr("q.id as question_id", "answer")
            .write.mode('overwrite').saveAsTable('training_dataset_answer'))

# COMMAND ----------

# MAGIC %sql select * from training_dataset_answer order by question_id desc

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM databricks_documentation d
# MAGIC   INNER JOIN training_dataset_question q on q.doc_id = d.id
# MAGIC   INNER JOIN training_dataset_answer   a on a.question_id = q.id 
