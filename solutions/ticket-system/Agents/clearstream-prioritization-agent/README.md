# LLM Ticket Prioritization Agent

This project implements a LangGraph-based agent for determining support ticket priorities. The agent runs on Databricks and uses the Mosaic AI Agent Framework.

## Project Structure

The project consists of three main files:

### 1. `config.yml`
Contains the configuration for the agent including:
- Agent prompt with priority levels and assessment criteria
- LLM endpoint configuration
- Example input format
- Warehouse ID

Example configuration:
```yaml
agent_prompt: |
  '"""Your task is to determine the priority of the ticket based on the following categories:
    Immediate: resolution expected in 2 to 4 hours...
    To assess the urgency:
    1. Nature of the Request...
    2. Impact on Client..."""'
llm_endpoint: "databricks-meta-llama-3-3-70b-instruct"
input_example:
  messages:
    - role: "user"
      content: "Your ticket text here..."
```

### 2. `agent.py`
Implements the core agent logic using LangGraph:
- MLflow setup and configuration
- Priority assessment processing
- JSON parsing and validation
- Output formatting

Key components:
```python
class JSONParser:
    """Custom parser for priority classification responses"""
    def validate_priority_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Validates priority and reasoning
```

### 3. `driver.py`
Handles deployment and evaluation:
- Logs the model to MLflow
- Runs evaluation on test examples
- Registers the model to Unity Catalog
- Deploys the model to a serving endpoint

## Setup and Usage

1. **Prerequisites**
   - Databricks runtime with required packages:
     ```
     langchain==0.2.16
     langchain-community==0.2.16
     langgraph-checkpoint==1.0.12
     langgraph==0.2.16
     pydantic
     databricks_langchain
     ```
   - Access to a Databricks workspace
   - Proper configurations in Unity Catalog

2. **Installation**
   - Copy the three files to your Databricks workspace
   - Update the `config.yml` with your specific settings
   - Ensure all dependencies are installed

3. **Running the Agent**
   ```python
   # Test priority classification
   response = agent.invoke({
       "messages": [{
           "role": "user",
           "content": "Your ticket text here"
       }]
   })
   ```

## Output Format

The agent returns priority assessments in the following format:
```json
{
    "priority": "Immediate",
    "priority_reasoning": "Explanation of priority decision based on assessment criteria..."
}
```

## Priority Levels

The agent supports four priority levels:
1. **Immediate**: 2-4 hours, before end of day
2. **Urgent**: 4-10 hours
3. **Normal**: 10-48 hours
4. **Low**: After 48 hours

## Assessment Criteria

The agent evaluates priorities based on:
1. Nature of the Request
2. Impact on Client
3. Tone of Voice
4. Context
5. Client Profile
6. Communication Frequency

## Development

To modify or extend the agent:
1. Edit the prompt in `config.yml` for priority criteria changes
2. Modify `agent.py` for assessment logic changes
3. Update `driver.py` for deployment changes

## Evaluation

The driver notebook includes evaluation capabilities:
- Validates priority assignments
- Checks reasoning completeness
- Logs results to MLflow

## Deployment

The agent can be deployed using:
```python
agents.deploy(
    UC_MODEL_NAME, 
    model_version, 
    tags={"endpointSource": "playground"}
)
```