# Databricks notebook source
# MAGIC %md
# MAGIC #Agent notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [**agent**]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [driver]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to recreate your agent from the AI Playground. It defines a LangChain agent that has access to the tools from the source Playground session.
# MAGIC
# MAGIC Use this notebook to iterate on and modify the agent. For example, you could add more tools or change the system prompt.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, however AI Agent Framework is compatible with other agent frameworks like Pyfunc and LlamaIndex.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After testing and iterating on your agent in this notebook, go to the auto-generated [driver]($./driver) notebook in this folder to log, register, and deploy the agent.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny langchain==0.2.16 langgraph-checkpoint==1.0.12 langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic databricks_langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import and setup
# MAGIC
# MAGIC Use `mlflow.langchain.autolog()` to set up [MLflow traces](https://docs.databricks.com/en/mlflow/mlflow-tracing.html).

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

mlflow.langchain.autolog()
config = ModelConfig(development_config="config.yml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the chat model and tools
# MAGIC Create a LangChain chat model that supports [LangGraph tool](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/) calling.
# MAGIC
# MAGIC Modify the tools your agent has access to by modifying the `uc_functions` list in [config.yml]($./config.yml). Any non-UC function spec tools can be defined in this notebook. See [LangChain - How to create tools](https://python.langchain.com/v0.2/docs/how_to/custom_tools/) and [LangChain - Using built-in tools](https://python.langchain.com/v0.2/docs/how_to/tools_builtin/).
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, however AI Agent Framework is compatible with other agent frameworks like Pyfunc and LlamaIndex.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

# Create the llm
llm = ChatDatabricks(endpoint=config.get("llm_endpoint"))

# We don't have a tool integration here so we can reduce complexity


# COMMAND ----------

# MAGIC %md
# MAGIC ## Output parsers
# MAGIC Databricks interfaces, such as the AI Playground, can optionally display pretty-printed tool calls.
# MAGIC
# MAGIC Use the following helper functions to parse the LLM's output into the expected format.

# COMMAND ----------

from typing import Iterator, Dict, Any
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    MessageLikeRepresentation,
    BaseMessage
)
import json


def stringify_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Convert a raw tool call into a formatted string that the playground UI expects if there is enough information in the tool_call
    """
    try:
        request = json.dumps(
            {
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("args", {})),
            },
            indent=2,
        )
        return f"<tool_call>{request}</tool_call>"
    except:
        return str(tool_call)


def stringify_tool_result(tool_msg: ToolMessage) -> str:
    """
    Convert a ToolMessage into a formatted string that the playground UI expects if there is enough information in the ToolMessage
    """
    try:
        result = json.dumps(
            {"id": tool_msg.tool_call_id, "content": tool_msg.content}, indent=2
        )
        return f"<tool_call_result>{result}</tool_call_result>"
    except:
        return str(tool_msg)


def parse_message(msg) -> str:
    """Parse different message types into their string representations"""
    # tool call result
    if isinstance(msg, ToolMessage):
        return stringify_tool_result(msg)
    # tool call
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_call_results = [stringify_tool_call(call) for call in msg.tool_calls]
        return "".join(tool_call_results)
    # normal HumanMessage or AIMessage (reasoning or final answer)
    elif isinstance(msg, (AIMessage, HumanMessage)):
        return msg.content
    else:
        print(f"Unexpected message type: {type(msg)}")
        return str(msg)


def wrap_output(stream: Iterator[MessageLikeRepresentation]) -> Iterator[str]:
    """
    Process and yield formatted outputs from the message stream.
    The invoke and stream langchain functions produce different output formats.
    This function handles both cases.
    """
    for event in stream:
        # the agent was called with invoke()
        if "messages" in event:
            # Only get the last message if it's an AI message
            messages = event["messages"]
            if messages and isinstance(messages[-1], AIMessage):
                yield parse_message(messages[-1]) + "\n\n"
        # the agent was called with stream()
        else:
            for node in event:
                for key, messages in event[node].items():
                    if isinstance(messages, list):
                        for msg in messages:
                            yield parse_message(msg) + "\n\n"
                    else:
                        print("Unexpected value {messages} for key {key}. Expected a list of `MessageLikeRepresentation`'s")
                        yield str(messages)


from typing import Any, Dict, Union
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


# Custom output parser for clearstream categories
class JSONParser:
    """Custom parser for ticket classification responses"""
    
    def parse_message(self, message: Union[BaseMessage, Dict, str]) -> Dict[str, Any]:
        """Parse different types of messages into JSON"""
        # If it's a Message object, get the content
        if isinstance(message, (AIMessage, HumanMessage)):
            content = message.content
        # If it's already a dict, return it
        elif isinstance(message, dict):
            if 'messages' in message:
                # If it's a message dict, get the last message's content
                messages = message['messages']
                if messages and isinstance(messages[-1], (AIMessage, HumanMessage)):
                    content = messages[-1].content
                else:
                    return {"error": "No valid message found in messages list"}
            else:
                # If it's already the format we want, return it
                return message
        # If it's a string, use it directly
        elif isinstance(message, str):
            content = message
        else:
            return {"error": f"Unexpected input type: {type(message)}"}
            
        # Now parse the content if it's a string
        if isinstance(content, str):
            try:
                return self.parse_json_string(content)
            except Exception as e:
                return {"error": f"Failed to parse content: {str(e)}"}
        
        return {"error": f"Unexpected content type: {type(content)}"}
    
    def parse_json_string(self, text: str) -> Dict[str, Any]:
        """Parse a string into JSON, handling edge cases"""
        import json
        
        try:
            # First try to parse the entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            try:
                # Find the first { and last }
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    json_str = text[start:end + 1]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            return {"error": "Could not parse JSON from text"}
    
    def validate_ticket_categories(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the parsed JSON has the expected ticket category fields"""
        expected_fields = {
            "service_category",
            "incident_category",
            "cause_category",
            "service_category_reasoning",
            "incident_category_reasoning",
            "cause_category_reasoning"
        }
        
        if not all(field in data for field in expected_fields):
            missing = expected_fields - set(data.keys())
            return {"error": f"Missing required fields: {missing}"}
            
        return data
    
    def __call__(self, input_data: Any) -> Dict[str, Any]:
        """Make the class callable and handle the parsing pipeline"""
        parsed = self.parse_message(input_data)
        if "error" not in parsed:
            return self.validate_ticket_categories(parsed)
        return parsed



# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the agent
# MAGIC Here we provide a simple graph that uses the model and tools defined by [config.yml]($./config.yml). This graph is adapated from [this LangGraph guide](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/).
# MAGIC
# MAGIC
# MAGIC To further customize your LangGraph agent, you can refer to:
# MAGIC * [LangGraph - Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/) for explanations of the concepts used in this LangGraph agent
# MAGIC * [LangGraph - How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/) to expand the functionality of your agent
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig
from typing import Iterator, Dict, Any, Annotated, Optional, Sequence, TypedDict, Union
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatDatabricks

# Setup MLflow logging
mlflow.langchain.autolog()
config = ModelConfig(development_config="config.yml")

# Create the LLM
llm = ChatDatabricks(endpoint=config.get("llm_endpoint"))

# Define the agent state
class AgentState(TypedDict):
    """The state of our agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent(model, agent_prompt: Optional[str] = None):
    """Create a simple agent that just processes messages with the system prompt"""
    
    def should_continue(state: AgentState):
        """Always end after one response since we don't have tools"""
        return "end"

    # Add system message if provided
    if agent_prompt:
        system_message = SystemMessage(content=agent_prompt)
        preprocessor = RunnableLambda(
            lambda state: [system_message] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    
    model_runnable = preprocessor | model

    def call_model(state: AgentState, config: RunnableConfig):
        """Call the model and return the response"""
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    # Create the workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.set_entry_point("agent")
    
    # Add edge to end since we don't have tools
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "end": END
        }
    )

    return workflow.compile()


# COMMAND ----------

from langchain_core.runnables import RunnableGenerator

# Create the agent with the system message if it exists
try:
    agent_prompt = config.get("agent_prompt")
    agent_with_raw_output = create_agent(llm, agent_prompt=agent_prompt)
except KeyError:
    print("No agent prompt found in config.yml. Skipping agent creation.")

# Add this if you want to try the agent in the Databricks Playground
agent = agent_with_raw_output | RunnableGenerator(wrap_output) 

# Use this output parsing if you want to return json only (e.g in a pipeline)
agent_json = agent_with_raw_output | RunnableGenerator(wrap_output) | JSONParser()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### JSON Ouput

# COMMAND ----------


# Use with invoke
#response = agent_json.invoke(config.get("input_example"))
#print(response)

# COMMAND ----------

# Or use with stream
#for event in agent_json.stream(config.get("input_example")):
#    print(event)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Playground compatible agent config

# COMMAND ----------


# Use with invoke
#response = agent.invoke(config.get("input_example"))
#print(response)

# COMMAND ----------

# Or use with stream
#for event in agent.stream(config.get("input_example")):
#    print(event)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Log model

# COMMAND ----------

# Enable when you want the playground compatible agent that can run with mlflow evaluation for testing
mlflow.models.set_model(agent)

# Enable when you want gent that is returning valid json (not compatible with playground and mlflow.evaluate)
#mlflow.models.set_model(agent_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC You can rerun the cells above to iterate and test the agent.
# MAGIC
# MAGIC Go to the auto-generated [driver]($./driver) notebook in this folder to log, register, and deploy the agent.
