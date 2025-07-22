from DQAgent import DQAgent
import json

# Initialize the DQ Agent
agent = DQAgent()

# Run data quality analysis on the orders table
# The agent will automatically retrieve 10 sample records using function calling
table_name = "orders"
response = agent.run(table_name)
print(json.dumps(response, indent=2))