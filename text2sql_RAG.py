import openai
import os
import faiss
import numpy as np
import json
from langchain_community.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModel
import torch

os.environ["OPENAI_API_TYPE"] = "azure" 
os.environ["OPENAI_API_KEY"] = "" #Can be found in your Azure Open AI Resource under Keys 
os.environ["OPENAI_API_BASE"] = "https://eastus.api.cognitive.microsoft.com/" #Can be found in your Azure Open AI Resource under Endpoint.
os.environ["openai_api_base"] = "https://eastus.api.cognitive.microsoft.com/"
os.environ["OPENAI_API_VERSION"] = "2024-06-01"
os.environ["OPENAI_DEPLOYMENT_NAME"] = "test2sql" #You can create a deployment of a Model within the Azure Open Ai studio - reference the name here. 
os.environ["OPENAI_MODEL_NAME"] = "gpt-35-turbo" #This is selected when creating the deployment of the Model 

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") 
llm = AzureOpenAI(deployment_name=os.environ.get('OPENAI_DEPLOYMENT_NAME'), model_name=os.environ.get('OPENAI_MODEL_NAME'), temperature=0)
prompt_template = PromptTemplate(
    input_variables=["input_text", "schema_info"],
    template="""
    You are a database expert. I will provide you with a question, and your job is to only provide the SQL query, nothing else. Do not explain anything.
    For Schema : {schema_info}

    Convert this user request into a SQL query:
    User Request: {input_text}

    SQL Query:
    """
)


def parse_json_schema(schema_data):
    schema_descriptions = []
    for table_name, table_info in schema_data.items():
        for column_name, column_info in table_info["columns"].items():
            column_description = f"Table: {table_name}, Column: {column_name}, Type: {column_info['type']}"
            schema_descriptions.append(column_description)
    return schema_descriptions

def generate_embeddings(text_list):
 
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the mean of the last hidden state across tokens for each sentence
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(sentence_embedding)
    return embeddings

def generate_embeddings_openai(text_list):
    response = openai.Embedding.create(
        input=text_list,
        engine="text-embedding-ada-002"
    )
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

def retrieve_relevant_schema(user_query):
    # Generate embedding for user query
    query_embedding = generate_embeddings([user_query])[0]
    
    # Convert query embedding to numpy array and search in FAISS
    query_embedding_np = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding_np, k=10)  # Retrieve top 3 schema descriptions
    
    # Return the relevant schema descriptions
    relevant_schema = [schema_lookup[i] for i in indices[0]]
    return relevant_schema

# Generate SQL Query
def text_to_sql(user_query):
    chain = LLMChain(llm=llm, prompt=prompt_template)
    # Retrieve relevant schema
    schema_context = retrieve_relevant_schema(user_query)
    schema_context_str = "\n".join(schema_context)
    
    # Generate SQL using the chain
    sql_query = chain.run(input_text=user_query, schema_info=schema_context_str)
    return sql_query

with open("db_schema.json", "r") as file:
    schema_data = json.load(file)
# Get schema descriptions
schema_descriptions = parse_json_schema(schema_data)
print("Schema Descriptions:", schema_descriptions)


# Generate embeddings for schema descriptions
schema_embeddings = generate_embeddings(schema_descriptions)

embedding_dim = len(schema_embeddings[0])  # Dimension of embeddings
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance metric

# Convert embeddings to numpy array and add to index
schema_embeddings_np = np.array(schema_embeddings).astype('float32')
index.add(schema_embeddings_np)

# Map index to schema descriptions
schema_lookup = {i: schema for i, schema in enumerate(schema_descriptions)}
user_query = "List all employee names who work in HR department and project name they are working on"
relevant_schema = retrieve_relevant_schema(user_query)
print("Relevant Schema:", relevant_schema)
# Example usage
generated_sql = text_to_sql(user_query)
print("Generated SQL Query:", generated_sql)
