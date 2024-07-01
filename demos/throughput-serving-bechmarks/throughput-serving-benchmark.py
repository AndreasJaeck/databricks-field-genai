# Databricks notebook source
n_gen_tokens = [293, 312, 449, 352, 423]
time = [8.54, 8.60, 9.50, 10.05, 11.25]

tokens_per_second = [tokens / t for tokens, t in zip(n_gen_tokens, time)]
total_tokens = sum(n_gen_tokens)
total_time = sum(time)
overall_tokens_per_second = total_tokens / total_time

print(tokens_per_second)
print(overall_tokens_per_second)

# COMMAND ----------

overall_tokens_per_second * 5


# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

model = ChatDatabricks(
    endpoint="dbrx-throughput-performance-test-aj"
)

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def invoke_concurrent_calls(num_calls, outer_iterations):
    overall_tokens_per_second = 0
    total_calls_made = 0

    def invoke_model():
        model = ChatDatabricks(endpoint="dbrx-throughput-performance-test-aj", max_tokens=2000)
        start_time = time.time()
        response = model.invoke("Write a text with 2000 tokens")
        end_time = time.time()
        runtime = end_time - start_time
        total_tokens = len(response.content.split())
        tokens_per_second = total_tokens / runtime if runtime > 0 else 0
        return runtime, total_tokens, tokens_per_second

    def outer_invoke():
        nonlocal overall_tokens_per_second
        nonlocal total_calls_made
        total_tokens_per_second = 0
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(invoke_model) for _ in range(num_calls)]
            for future in as_completed(futures):
                runtime, total_tokens, tokens_per_second = future.result()
                total_tokens_per_second += tokens_per_second
                total_calls_made += 1
                print(f"Runtime: {runtime}s, Total Tokens: {total_tokens}, Tokens/Second: {tokens_per_second}")
        
        overall_tokens_per_second += total_tokens_per_second

    with ThreadPoolExecutor(max_workers=outer_iterations) as executor:
        futures = [executor.submit(outer_invoke) for _ in range(outer_iterations)]
        for future in as_completed(futures):
            future.result()
            time.sleep(2)

    overall_average_tokens_per_second = overall_tokens_per_second / total_calls_made if total_calls_made > 0 else 0
    print(f"Overall Average Tokens/Second: {overall_average_tokens_per_second}")

invoke_concurrent_calls(8, 10)

# COMMAND ----------

10 * 16

# COMMAND ----------

Runtime: 65.8693835735321s, Total Tokens: 392, Tokens/Second: 5.951171526637984
Runtime: 72.82459712028503s, Total Tokens: 465, Tokens/Second: 6.3852052519007465
Runtime: 80.10507583618164s, Total Tokens: 533, Tokens/Second: 6.653760631723364
Runtime: 81.7260365486145s, Total Tokens: 546, Tokens/Second: 6.680857448350789
Runtime: 81.74114322662354s, Total Tokens: 547, Tokens/Second: 6.691856492433288
Runtime: 81.72402119636536s, Total Tokens: 546, Tokens/Second: 6.681022201392644
Runtime: 84.88165831565857s, Total Tokens: 557, Tokens/Second: 6.562077262070259
Runtime: 86.00795388221741s, Total Tokens: 564, Tokens/Second: 6.5575330483081045
Runtime: 87.23108625411987s, Total Tokens: 576, Tokens/Second: 6.603150605301511
Runtime: 87.31357741355896s, Total Tokens: 561, Tokens/Second: 6.425117566112714
Runtime: 88.71873927116394s, Total Tokens: 589, Tokens/Second: 6.638958182213951
Runtime: 88.69745945930481s, Total Tokens: 589, Tokens/Second: 6.6405509649375976
Runtime: 89.74920082092285s, Total Tokens: 592, Tokens/Second: 6.596159014064329
Runtime: 90.43843507766724s, Total Tokens: 589, Tokens/Second: 6.512717734381131
Runtime: 90.30130386352539s, Total Tokens: 589, Tokens/Second: 6.522607922585153
Runtime: 90.41580748558044s, Total Tokens: 589, Tokens/Second: 6.514347616636991
Runtime: 93.08312153816223s, Total Tokens: 580, Tokens/Second: 6.230990005660817
Runtime: 93.88619351387024s, Total Tokens: 612, Tokens/Second: 6.518530330123421
Runtime: 94.27104496955872s, Total Tokens: 615, Tokens/Second: 6.523742260400223
Runtime: 94.58461213111877s, Total Tokens: 616, Tokens/Second: 6.512687276721762
Runtime: 95.27390193939209s, Total Tokens: 619, Tokens/Second: 6.497057299004853
Runtime: 96.05661582946777s, Total Tokens: 611, Tokens/Second: 6.360832043934661
Runtime: 96.05001473426819s, Total Tokens: 611, Tokens/Second: 6.3612691959537075
Runtime: 97.05358910560608s, Total Tokens: 623, Tokens/Second: 6.419134065429568
Runtime: 98.7294499874115s, Total Tokens: 637, Tokens/Second: 6.451975576499421
Runtime: 101.09439468383789s, Total Tokens: 662, Tokens/Second: 6.548335365875976
Runtime: 101.2155532836914s, Total Tokens: 662, Tokens/Second: 6.54049677666156
Runtime: 102.13957905769348s, Total Tokens: 666, Tokens/Second: 6.520488983255064
Runtime: 102.92912077903748s, Total Tokens: 659, Tokens/Second: 6.402464093856437
Runtime: 103.67283844947815s, Total Tokens: 664, Tokens/Second: 6.40476338769851
Runtime: 103.71237683296204s, Total Tokens: 672, Tokens/Second: 6.479458098644441
Runtime: 105.73208260536194s, Total Tokens: 686, Tokens/Second: 6.488096924756982


# COMMAND ----------

model.invoke("Write a text with 1000 tokens")

# COMMAND ----------


