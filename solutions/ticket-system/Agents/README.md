# Clearstream Support Ticket Processing Agents

This repository contains three LLM-based agents for processing support tickets. Each agent is built using the LangGraph framework and runs on Databricks using the Mosaic AI Agent Framework.

## Overview

The project consists of three specialized agents:

### 1. Ticket Classification Agent
- Classifies tickets into a three-level category hierarchy
- Service Categories (e.g., Cash, Corporate Action)
- Incident Categories (e.g., New Instruction, Suspense)
- Cause Categories (e.g., Deadline, Formats)
- [Detailed Documentation](./clearstream-categorization-agent/README.md)

### 2. Information Extraction Agent
- Extracts structured information from ticket text
- Handles various field types (IDs, dates, ISINs, etc.)
- Validates field formats
- Flexible field presence handling
- [Detailed Documentation](./clearstream-extraction-agent/README.md)

### 3. Priority Assessment Agent
- Determines ticket urgency levels
- Four priority levels (Immediate to Low)
- Considers multiple assessment criteria
- Provides reasoning for priority decisions
- [Detailed Documentation](./clearstream-prioritiyation-agent/README.md)

## Project Structure

Each agent follows the same basic structure:
```
├── agent_name/
│   ├── README.md         # Agent-specific documentation
│   ├── agent.py          # Core agent implementation
│   ├── config.yml        # Agent configuration
│   └── driver.py         # Deployment and evaluation
```

## Common Dependencies

All agents require:
```
langchain==0.2.16
langchain-community==0.2.16
langgraph-checkpoint==1.0.12
langgraph==0.2.16
pydantic
databricks_langchain
```

## Quick Start

1. Choose the agent you want to use
2. Copy its files to your Databricks workspace
3. Update the config.yml with your settings
4. Run the driver notebook to deploy

Notebook Example usage:
```python
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Your ticket text here"
    }]
})
```

## Integration Options

Each agent comes in two variants:
1. **agent_json**: Returns pure JSON output
   - Suitable for system integration and programmatic use
   - Can be called from other applications
   - Returns valid JSON objects directly
   - Example:
     ```python
     response = agent_json.invoke(ticket_text)
     # Returns: {"priority": "Immediate", "priority_reasoning": "..."}
     ```

2. **agent**: Returns JSON wrapped in markdown code blocks
   - Required for MLflow evaluation
   - Compatible with Databricks Playground
   - Returns JSON as ```json wrapped strings
   - Example:
     ```python
     response = agent.invoke(ticket_text)
     # Returns: ```json\n{"priority": "Immediate", "priority_reasoning": "..."}\n```
     ```

### Endpoint Integration

Once deployed, the agents can be accessed via REST API endpoints. Here are examples of calling the endpoints:

1. **Using curl:**
```bash
curl \
  -u token:$DATABRICKS_TOKEN \
  -X POST \
  -H "Content-Type: application/json" \
  -d@data.json \
  https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint-name/invocations
```

2. **Using Python:**
```python
import os
import requests
import json
import pandas as pd

def score_model(dataset):
    url = 'https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint-name/invocations'
    headers = {
        'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        'Content-Type': 'application/json'
    }
    
    # Handle both DataFrame and dictionary inputs
    ds_dict = {
        'dataframe_split': dataset.to_dict(orient='split')
    } if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(
        method='POST',
        headers=headers,
        url=url,
        data=data_json
    )
    
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
    return response.json()

def create_tf_serving_json(data):
    return {
        'inputs': {name: data[name].tolist() for name in data.keys()} 
        if isinstance(data, dict) else data.tolist()
    }
```

### Pipeline Integration

There are two ways to integrate the agents in a pipeline:

#### 1. Direct Endpoint Calls

Call each agent endpoint in sequence:

```python
def process_ticket(ticket_data):
    # Call each endpoint in sequence
    info = score_model(ticket_data, extraction_endpoint)
    categories = score_model(ticket_data, classification_endpoint)
    priority = score_model(ticket_data, priority_endpoint)
    
    return {**info, **categories, **priority}

# Process a batch of tickets
ticket_batch = pd.DataFrame({
    "messages": [{"role": "user", "content": text} for text in ticket_texts]
})
results = process_ticket(ticket_batch)
```

Note: Ensure you have:
- Set your DATABRICKS_TOKEN environment variable
- Proper endpoint URLs for your workspace
- Appropriate authentication and access rights
- Correct input data format for each endpoint

#### 2. MLflow PythonModel Pipeline

For a more integrated approach, you can wrap all three agents in a single MLflow model:

```python
class TicketProcessingPipeline(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.extraction_endpoint = None
        self.classification_endpoint = None
        self.priority_endpoint = None
    
    def load_context(self, context):
        """Load endpoint configurations during model loading"""
        # Load endpoints from context
        self.extraction_endpoint = "your-extraction-endpoint"
        self.classification_endpoint = "your-classification-endpoint"
        self.priority_endpoint = "your-priority-endpoint"
    
    def predict(self, context, model_input: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Process tickets through all three agents"""
        results = []
        
        for ticket_text in model_input['content']:
            # Call each endpoint
            ticket_data = {"messages": [{"role": "user", "content": ticket_text}]}
            
            # Extract information
            info = score_model(ticket_data, self.extraction_endpoint)
            
            # Classify ticket
            categories = score_model(ticket_data, self.classification_endpoint)
            
            # Determine priority
            priority = score_model(ticket_data, self.priority_endpoint)
            
            # Combine results
            result = {**info, **categories, **priority}
            results.append(result)
        
        return pd.DataFrame(results)

# Log the pipeline model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "ticket_pipeline",
        python_model=TicketProcessingPipeline(),
        pip_requirements=["pandas", "requests"]
    )
```

Benefits of using MLflow PythonModel:
1. **Encapsulation**: All agent interactions are wrapped in a single model
2. **State Management**: Endpoint configurations can be stored and loaded
3. **Batch Processing**: Efficient handling of multiple tickets
4. **Deployment**: Can be deployed as a single endpoint
5. **Version Control**: Pipeline versions can be tracked in MLflow
6. **Dependencies**: Required packages are explicitly specified
7. **Flexibility**: Custom preprocessing and post-processing can be added

The MLflow pipeline can be deployed as a single endpoint, simplifying integration into existing systems while maintaining all the benefits of MLflow's model management capabilities.

## Author Notes

### Performance and Cost Considerations

1. **Large Prompt Impact:**
   - Some agents use extensive prompts which can affect:
     - Response times
     - Token usage costs, especially with pay-per-token endpoints
   - Mitigation strategies:
     - Consider using LLM vendors with prompt caching
     - During development, work with a smaller ticket subset first
     - Use sample size to estimate costs for full dataset (400k tickets)
     - Evaluate if fine-tuning could be more cost-effective

2. **Improvement Strategies:**
   - Easiest performance gains come from:
     - Prompt engineering and refinement
     - Adding specific examples for categorization
     - Including domain-specific prioritization examples
   - Consider maintaining a dedicated example set for each agent

## Development and Deployment

1. **Testing:**
   - Each agent includes evaluation capabilities
   - Use test cases in driver notebooks
   - Monitor MLflow metrics

2. **Deployment:**
   - Use driver notebooks for:
     - Model logging
     - Performance evaluation
     - Unity Catalog registration
     - Endpoint deployment

## Monitoring and Maintenance

Monitor:
- Response times
- Token usage
- Classification accuracy
- Extraction quality
- Priority assignment appropriateness

## Support and Documentation

- Individual agent READMEs contain detailed information
- Config files document expected inputs/outputs
- Driver notebooks include usage examples
