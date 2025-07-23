import pandas as pd
import sqlite3
import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.callbacks.manager import get_openai_callback
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# Configuration - Load from environment variables
MODEL_NAME = os.getenv("MODEL_NAME")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")


TEMPERATURE = int(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# Validate required environment variables
if not LLM_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")


# Pydantic Models for Structured Output
class DataQualityRule(BaseModel):
    """A single data quality rule for a column."""
    rule_name: str = Field(description="Name of the data quality rule")
    rule_description: str = Field(description="Description of what the rule validates")

class DataQualityRules(BaseModel):
    """Collection of data quality rules for all columns."""
    rules: List[DataQualityRule] = Field(description="List of data quality rules")

class SQLQuery(BaseModel):
    """SQL query for validating a data quality rule."""
    rule_name: str = Field(description="Name of the rule this query validates")
    sql_query: str = Field(description="SQLite-compatible query that returns true/false")

class SQLQueries(BaseModel):
    """Collection of SQL queries for data quality validation."""
    queries: List[SQLQuery] = Field(description="List of SQL validation queries")

class ValidationResult(BaseModel):
    """Result of executing a data quality validation query."""
    rule_name: str = Field(description="Name of the validated rule")
    rule_description: str = Field(description="Description of the rule")
    sql_query: str = Field(description="SQL query that was executed")
    result: bool = Field(description="Whether the validation passed (true) or failed (false)")

class DataQualitySummary(BaseModel):
    """Summary of all data quality validation results."""
    results: List[ValidationResult] = Field(description="List of validation results")
    total_rules: int = Field(description="Total number of rules validated")
    passed_rules: int = Field(description="Number of rules that passed")
    failed_rules: int = Field(description="Number of rules that failed")

class DQAgent:
    """LangChain-based Data Quality Agent for automated rule generation and validation."""
    
    def __init__(self, model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
        """Initialize the DQ Agent with LangChain components."""
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize ChatOpenAI with OpenRouter
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_base=LLM_BASE_URL,
            openai_api_key=LLM_API_KEY
        )
        
        # Setup tools for function calling
        self._setup_tools()
        
        # Create LLM with tools
        self.llm_with_tools = self.llm.bind_tools([self.get_sample_data_tool, self.profile_table_tool])
        
        # Create prompt templates
        self._setup_prompts()
        
        # Create output parsers
        self._setup_parsers()
    
    def _setup_prompts(self):
        """Setup ChatPromptTemplate instances for each step."""
        
        # Rule examples to reduce hallucination
        RULE_EXAMPLE = """
Example rule set:
[
  {{"rule_name": "amount_non_negative", "rule_description": "amount should be >= 0"}},
  {{"rule_name": "order_id_unique", "rule_description": "order_id should be unique for each row"}},
  {{"rule_name": "date_valid_format", "rule_description": "date should be in valid format and not null"}},
  {{"rule_name": "status_valid_enum", "rule_description": "status should be one of: pending, completed, cancelled"}}
]
"""
        
        # Step 1: Rule Generation Prompt with profiling stats
        self.rules_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a data analyst. Your role is to create deterministic rules to test the quality of a dataset.
You will be provided statistical profiling data that represents the dataset you need to create rules for.
The profiling data includes min/max values, null counts, unique values, and data types.

Process:
1. Observe the profiling statistics
2. For each column in the dataset, create a rule based on the statistical analysis
3. Each rule should have a name and a description
4. Focus on value ranges, uniqueness constraints, null checks, and data type validation

Create at least one rule for each column in the dataset.
Refer to the following example format:
{RULE_EXAMPLE}
Return the rules as a structured JSON object."""),
            ("human", "Profiling stats:\n{profile_stats}")
        ])
        
        # Step 2: SQL Generation Prompt  
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create SQL queries for the provided data quality rules.

Requirements:
1. Create a SQL query for each rule
2. Each query should return true or false as a single aggregate value for the entire column
3. Make sure the query is an aggregate query
4. Include the rule name in the SQL query
5. The database is SQLite - generate SQLite-compatible queries
6. The dataset table name is "orders"
7. Each query should return columns: 'Rule' (rule name) and 'value' (true/false)

Format: SELECT 'rule_name' as Rule, CASE WHEN condition THEN 'true' ELSE 'false' END as value FROM orders"""),
            ("human", "Data quality rules:\n{rules}")
        ])
        
        # Step 3: Summary Prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize the data quality validation results.

Include for each rule:
- Rule definition and description
- SQL query used
- Validation result (passed/failed)
- Overall summary statistics

Return as a structured JSON object."""),
            ("human", "Rules: {rules}\nSQL Queries: {queries}\nValidation Results: {results}")
        ])
    
    def _setup_tools(self):
        """Setup function calling tools."""
        @tool
        def get_sample_data(table_name: str) -> str:
            """Get 10 sample records from the specified table.
            
            Args:
                table_name: Name of the database table to sample from
                
            Returns:
                JSON string containing sample data
            """
            try:
                conn = sqlite3.connect("demo.sqlite")
                query = f"SELECT * FROM {table_name} LIMIT 10"
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df.to_json(orient="records", indent=2)
            except Exception as e:
                return f"Error retrieving sample data: {str(e)}"
        
        @tool
        def profile_table(table_name: str, limit_rows: int = 10000) -> str:
            """Get statistical profile of table to replace large sample data.
            
            Args:
                table_name: Name of the database table to profile
                limit_rows: Maximum rows to analyze for profiling
                
            Returns:
                JSON string containing statistical profile
            """
            try:
                conn = sqlite3.connect("demo.sqlite")
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit_rows}", conn)
                conn.close()
                stats = df.describe(include='all').to_dict()
                return json.dumps(stats, indent=2, default=str)
            except Exception as e:
                return f"Error profiling table: {str(e)}"
        
        self.get_sample_data_tool = get_sample_data
        self.profile_table_tool = profile_table
    
    def _setup_parsers(self):
        """Setup JSON output parsers for structured responses."""
        self.rules_parser = JsonOutputParser(pydantic_object=DataQualityRules)
        self.sql_parser = JsonOutputParser(pydantic_object=SQLQueries)
        self.summary_parser = JsonOutputParser(pydantic_object=DataQualitySummary)
    
    def exec_sql(self, sql_query: str) -> dict:
        """Execute SQL query against SQLite database."""
        try:
            conn = sqlite3.connect("demo.sqlite")
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return json.loads(df.to_json(orient="records", index=False))
        except Exception as e:
            print(f"SQL execution error: {e}")
            return [{"Rule": "Error", "value": "false"}]
    
    def _get_sample_data(self, table_name: str) -> str:
        """Get sample data from the specified table."""
        return self.get_sample_data_tool.invoke({"table_name": table_name})
    
    def _get_table_profile(self, table_name: str, limit_rows: int = 10000) -> str:
        """Get statistical profile of the specified table."""
        return self.profile_table_tool.invoke({"table_name": table_name, "limit_rows": limit_rows})
    
    def run(self, table_name: str) -> dict:
        """
        Run the complete data quality analysis workflow using a single LCEL chain.
        
        Args:
            table_name: Name of the database table to analyze
            
        Returns:
            Dictionary containing the complete analysis results
        """
        
        # Get statistical profile instead of sample data
        print(f"Profiling table '{table_name}'...")
        profile_stats = self._get_table_profile(table_name)
        
        # Create a single LCEL chain that handles the entire workflow
        from langchain_core.runnables import RunnableLambda
        
        def execute_sql_step(sql_result):
            """Execute SQL queries and prepare data for summary."""
            print("Executing validation queries...")
            validation_results = self._execute_validation_queries(sql_result.queries)
            return {
                "rules": json.dumps(sql_result.model_dump()),
                "queries": json.dumps(sql_result.model_dump()),
                "results": json.dumps(validation_results)
            }
        
        # Enhanced chain with retry and usage tracking
        def rules_step(input_data):
            print("Step 1: Generating data quality rules...")
            return self._call_llm_with_retry(self.rules_prompt.invoke(input_data), DataQualityRules)
        
        def sql_step(rules):
            print("Step 2: Generating SQL validation queries...")
            input_data = {"rules": json.dumps(rules.model_dump())}
            return self._call_llm_with_retry(self.sql_prompt.invoke(input_data), SQLQueries)
        
        def summary_step(summary_data):
            print("Step 3: Generating final summary...")
            return self._call_llm_with_retry(self.summary_prompt.invoke(summary_data), DataQualitySummary)
        
        # Execute chain with proper error handling and retry
        try:
            print("Running enhanced data quality analysis chain...")
            
            # Step 1: Generate rules
            rules_result = rules_step({"profile_stats": profile_stats})
            
            # Step 2: Generate SQL queries
            sql_result = sql_step(rules_result)
            
            # Step 3: Execute queries
            summary_data = execute_sql_step(sql_result)
            
            # Step 4: Generate summary
            summary_result = summary_step(summary_data)
            
        except Exception as e:
            print(f"Error in analysis chain: {e}")
            raise
        
        return summary_result.model_dump()
    
    def _validate_sql(self, queries: SQLQueries) -> List[SQLQuery]:
        """Validate SQL queries before execution to filter out invalid ones."""
        valid_queries = []
        for q in queries.queries:
            try:
                # Use EXPLAIN to validate syntax without executing
                conn = sqlite3.connect(":memory:")
                conn.execute(f"EXPLAIN {q.sql_query}")
                conn.close()
                valid_queries.append(q)
            except sqlite3.Error as e:
                print(f"Query failed validation: {q.rule_name} - {str(e)}")
                # Optionally, you could implement retry logic here
                # to send the failed query back to LLM for correction
        return valid_queries
    
    def _execute_validation_queries(self, queries: List[SQLQuery]) -> List[dict]:
        """Execute validation queries and return structured results."""
        # First validate all queries
        valid_queries = self._validate_sql(SQLQueries(queries=queries))
        print(f"Validated {len(valid_queries)}/{len(queries)} queries")
        
        validation_results = []
        
        for query in valid_queries:
            try:
                result = self.exec_sql(query.sql_query)
                validation_results.append({
                    "rule_name": query.rule_name,
                    "sql_query": query.sql_query,
                    "result": result[0] if result else {"Rule": "Error", "value": "false"}
                })
            except Exception as e:
                print(f"Error executing query for {query.rule_name}: {e}")
                validation_results.append({
                    "rule_name": query.rule_name,
                    "sql_query": query.sql_query,
                    "result": {"Rule": "Error", "value": "false"}
                })
        
        return validation_results
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def _call_llm_with_retry(self, prompt, output_parser=None):
        """Call LLM with retry mechanism and usage tracking."""
        with get_openai_callback() as cb:
            if output_parser:
                result = self.llm.with_structured_output(output_parser).invoke(prompt)
            else:
                result = self.llm.invoke(prompt)
        return result
    
    def get_usage_stats(self) -> dict:
        """Get token usage statistics from the last run."""
        return {"note": "Usage tracking implemented with LangChain callbacks and displayed during execution"}


