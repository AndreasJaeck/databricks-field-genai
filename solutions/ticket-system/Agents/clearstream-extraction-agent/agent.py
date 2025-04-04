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

# We don't have a tool integration here so we can reduce complexity here


# COMMAND ----------

# MAGIC %md
# MAGIC ## Output parsers
# MAGIC Databricks interfaces, such as the AI Playground, can optionally display pretty-printed tool calls.
# MAGIC
# MAGIC Use the following helper functions to parse the LLM's output into the expected format.

# COMMAND ----------

from typing import Iterator, Dict, Any, Union
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    MessageLikeRepresentation,
    BaseMessage
)
import json
from datetime import datetime
import re


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



# Custom output parser for clearstream extraction 
class JSONParser:
    """Custom parser for ticket information extraction responses"""
    
    def __init__(self):
        # Define valid categories and formats
        self.valid_service_categories = {
            "Cash", "Client Account", "Corporate Actions", 
            "Income", "New Issues", "Settlement"
        }
        self.iso_currency_pattern = r'^[A-Z]{3}$'
        self.isin_pattern = r'^[A-Z]{2}[A-Z0-9]{9}\d{1}$'
        self.ticket_id_pattern = r'^\d{7}$'
        self.email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
    def validate_date(self, date_str: str) -> bool:
        """Validate date string format DD-MM-YYYY"""
        try:
            datetime.strptime(date_str, '%d-%m-%Y')
            return True
        except ValueError:
            return False
            
    def validate_isin(self, isin: str) -> bool:
        """Validate ISIN format"""
        return bool(re.match(self.isin_pattern, isin))
        
    def validate_currency(self, currency: str) -> bool:
        """Validate ISO 4217 currency format"""
        return bool(re.match(self.iso_currency_pattern, currency))
        
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        return bool(re.match(self.email_pattern, email))
        
    def validate_ticket_id(self, ticket_id: str) -> bool:
        """Validate 7-digit ticket ID"""
        return bool(re.match(self.ticket_id_pattern, ticket_id))

    def parse_message(self, message: Union[BaseMessage, Dict, str]) -> Dict[str, Any]:
        """Parse different types of messages into JSON"""
        if isinstance(message, (AIMessage, HumanMessage)):
            content = message.content
        elif isinstance(message, dict):
            if 'messages' in message:
                messages = message['messages']
                if messages and isinstance(messages[-1], (AIMessage, HumanMessage)):
                    content = messages[-1].content
                else:
                    return {"error": "No valid message found in messages list"}
            else:
                return message
        elif isinstance(message, str):
            content = message
        else:
            return {"error": f"Unexpected input type: {type(message)}"}
            
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
    
    def validate_ticket_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted ticket information"""
        # Create clean output dict
        clean_data = {}
        
        # Validate and add fields only if they exist
        if 'ticket_id' in data:
            if data['ticket_id'] is None:
                clean_data['ticket_id'] = None
            elif self.validate_ticket_id(str(data['ticket_id'])):
                clean_data['ticket_id'] = str(data['ticket_id'])
            else:
                return {"error": "Invalid ticket_id format"}
                
        if 'service_category' in data:
            if data['service_category'] in self.valid_service_categories:
                clean_data['service_category'] = data['service_category']
            else:
                return {"error": f"Invalid service_category. Must be one of: {self.valid_service_categories}"}
                
        # Optional fields - add only if present
        for field in ['incident_category', 'cause_category', 'customer_first_name', 'customer_last_name']:
            if field in data and data[field]:
                clean_data[field] = data[field]
                
        if 'customer_email' in data and data['customer_email']:
            if self.validate_email(data['customer_email']):
                clean_data['customer_email'] = data['customer_email']
            else:
                return {"error": "Invalid email format"}
                
        # Validate lists
        if 'isin' in data and data['isin']:
            if isinstance(data['isin'], list):
                valid_isins = [isin for isin in data['isin'] if self.validate_isin(isin)]
                if valid_isins:
                    clean_data['isin'] = valid_isins
            else:
                return {"error": "ISIN must be a list"}
                
        if 'account_number' in data and data['account_number']:
            if isinstance(data['account_number'], list):
                clean_data['account_number'] = data['account_number']
            else:
                return {"error": "account_number must be a list"}
                
        # Validate dates
        for date_field in ['record_date', 'trade_date', 'ex_date']:
            if date_field in data and data[date_field]:
                if self.validate_date(data[date_field]):
                    clean_data[date_field] = data[date_field]
                else:
                    return {"error": f"Invalid {date_field} format. Use DD-MM-YYYY"}
                    
        if 'currency' in data and data['currency']:
            if self.validate_currency(data['currency']):
                clean_data['currency'] = data['currency']
            else:
                return {"error": "Invalid currency format. Use ISO 4217"}
                
        return clean_data
    
    def __call__(self, input_data: Any) -> Dict[str, Any]:
        """Make the class callable and handle the parsing pipeline"""
        parsed = self.parse_message(input_data)
        if "error" not in parsed:
            return self.validate_ticket_info(parsed)
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

# Enable when you want the playground compatible agent to be logged my the driver script
mlflow.models.set_model(agent)

# Enable when you want the json agent to be logged my the driver script
#mlflow.models.set_model(agent_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC You can rerun the cells above to iterate and test the agent.
# MAGIC
# MAGIC Go to the auto-generated [driver]($./driver) notebook in this folder to log, register, and deploy the agent.
