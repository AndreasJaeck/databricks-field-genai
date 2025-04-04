# LLM Ticket Classification Agent

This project implements a LangGraph-based agent for classifying support tickets into predefined categories. The agent runs on Databricks and uses the Mosaic AI Agent Framework.

## Project Structure

The project consists of three main files:

### 1. `config.yml`
Contains the configuration for the agent including:
- Agent prompt with the category hierarchy
- LLM endpoint configuration
- Example input format
- Warehouse ID

Example configuration:
```yaml
agent_prompt: |
  '"""Consider the following structure..."""'
llm_endpoint: "databricks-meta-llama-3-3-70b-instruct"
warehouse_id: "85eca9d9e2ed6de3"
input_example:
  messages:
    - role: "user"
      content: "Your ticket text here..."
```

### 2. `agent.py`
Implements the core agent logic using LangGraph:
- MLflow setup and configuration
- Agent state management
- Message processing pipeline
- Output formatting

Key components:
```python
# Create the agent
agent_with_raw_output = create_agent(llm, agent_prompt=agent_prompt)
agent = agent_with_raw_output | RunnableGenerator(wrap_output)
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
   # Test a single classification
   response = agent.invoke({
       "messages": [{
           "role": "user",
           "content": "Your ticket text here"
       }]
   })
   ```

4. **Deployment**
   - Run the driver notebook to:
     - Log the model
     - Evaluate performance
     - Deploy to production

## Output Format

The agent returns classifications in the following format:
```json
{
    "service_category": "Category name",
    "incident_category": "Subcategory name",
    "cause_category": "Specific cause",
    "service_category_reasoning": "Explanation...",
    "incident_category_reasoning": "Explanation...",
    "cause_category_reasoning": "Explanation..."
}
```

## Category Structure

The agent supports a three-level category hierarchy:
1. Service Categories (e.g., Cash, Corporate Action)
2. Incident Categories (e.g., New Instruction, Suspense)
3. Cause Categories (e.g., Deadline, Formats)

See `config.yml` for the complete category structure.

## Development

To modify or extend the agent:
1. Edit the prompt in `config.yml` for category changes
2. Modify `agent.py` for processing logic changes
3. Update `driver.py` for deployment changes

## Evaluation

The driver notebook includes evaluation capabilities:
- Uses predefined test cases
- Computes accuracy metrics
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