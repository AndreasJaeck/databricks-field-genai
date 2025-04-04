# LLM Ticket Information Extraction Agent

This project implements a LangGraph-based agent for extracting structured information from support tickets. The agent runs on Databricks and uses the Mosaic AI Agent Framework.

## Project Structure

The project consists of three main files:

### 1. `config.yml`
Contains the configuration for the agent including:
- Agent prompt specifying extraction fields and formats
- LLM endpoint configuration
- Example input format
- Warehouse ID

Example configuration:
```yaml
agent_prompt: |
  '"""You are an information extractor. Your task:
    Input: Text blocks containing ticket information
    Possible fields (return keys only if value is present):..."""'
llm_endpoint: "databricks-meta-llama-3-3-70b-instruct"
input_example:
  messages:
    - role: "user"
      content: "Your ticket text here..."
```

### 2. `agent.py`
Implements the core agent logic using LangGraph:
- MLflow setup and configuration
- JSON parsing and validation
- Field format validation (ISIN, dates, email, etc.)
- Output formatting

Key components:
```python
class JSONParser:
    """Custom parser for ticket information extraction responses"""
    def validate_ticket_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Validates and cleans extracted information
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
   # Test information extraction
   response = agent.invoke({
       "messages": [{
           "role": "user",
           "content": "Your ticket text here"
       }]
   })
   ```

## Output Format

The agent extracts information in the following format:
```json
{
    "ticket_id": "1234567",
    "service_category": "Cash",
    "incident_category": "Optional incident type",
    "cause_category": "Optional cause",
    "customer_first_name": "John",
    "customer_last_name": "Smith",
    "customer_email": "john.smith@example.com",
    "isin": ["US5949181045"],
    "account_number": ["ABC123"],
    "record_date": "31-01-2024",
    "trade_date": "31-01-2024",
    "ex_date": "31-01-2024",
    "currency": "USD"
}
```

Note: Fields are only included if present in the input text and valid.

## Field Specifications

The agent extracts the following fields:
1. **Required Format Fields**
   - ticket_id: 7-digit number or null
   - service_category: Predefined list of categories
   - currency: ISO 4217 format
   - dates: DD-MM-YYYY format
   - email: Valid email format
   - ISIN: Standard ISIN format

2. **Free Text Fields**
   - incident_category
   - cause_category
   - customer names

3. **List Fields**
   - isin: List of valid ISIN codes
   - account_number: List of account numbers

## Development

To modify or extend the agent:
1. Edit the prompt in `config.yml` for field changes
2. Modify `agent.py` for validation logic changes
3. Update `driver.py` for deployment changes

## Evaluation

The driver notebook includes evaluation capabilities:
- Validates field formats
- Checks extraction accuracy
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