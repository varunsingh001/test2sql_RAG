import os, faiss, json
import numpy as np
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer
import numpy as np


# Configure logging
#logging.basicConfig(level=logging.DEBUG)

#azure_logger = logging.getLogger("azure")
#azure_logger.setLevel(logging.DEBUG)

load_dotenv()

credential=DefaultAzureCredential()
print("Using credential:", credential)

# Authenticate using DefaultAzureCredential
secret_client = SecretClient(vault_url=os.environ.get('VAULT_URI'), credential=credential)
os.environ["OPENAI_API_KEY"] = secret_client.get_secret("OpenAI-Api-Key").value

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm = AzureChatOpenAI(deployment_name=os.environ.get('OPENAI_DEPLOYMENT_NAME'), model_name=os.environ.get('OPENAI_MODEL_NAME'), temperature=0)


def generate_embeddings_locally(text_list):
    embeddings = model.encode(text_list)
    return embeddings

def process_schema():
    with open("RAG_Experiments/text2sql/schema.json", "r") as file:
        schema_data = json.load(file)
    # Get schema descriptions
    schema_descriptions = parse_json_schema(schema_data)
    print("Schema Descriptions:", schema_descriptions)
    # Generate embeddings for schema descriptions
    schema_embeddings = generate_embeddings_locally(schema_descriptions)
    embedding_dim = len(schema_embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance metric
    # Convert embeddings to numpy array and add to index
    schema_embeddings_np = np.array(schema_embeddings).astype('float32')
    index.add(schema_embeddings_np)
    return (index, schema_descriptions)

def parse_json_schema(schema_data):
    schema_descriptions = []
    for table_name, table_info in schema_data.items():
        for column_name, column_info in table_info["columns"].items():
            column_description = f"Table: {table_name}, Column: {column_name}, Type: {column_info['type']}"
            schema_descriptions.append(column_description)
        for column_name in table_info["primary_key"].items():
            column_description = f"Table: {table_name}, Column: {column_name}, Type: {column_info['type']}"
            schema_descriptions.append(column_description)
    return schema_descriptions
                                        
def retrieve_relevant_schema(user_query, schema_data):
    # Generate embedding for user query
    query_embedding = generate_embeddings_locally([user_query])[0]
    # Convert query embedding to numpy array and search in FAISS
    query_embedding_np = np.array([query_embedding]).astype('float32')
    distances, indices = schema_data[0].search(query_embedding_np, k=10)  # Retrieve top 3 schema descriptions
    # Return the relevant schema descriptions
    schema_lookup = {i: schema for i, schema in enumerate(schema_data[1])}
    relevant_schema = [schema_lookup[i] for i in indices[0]]
    return relevant_schema

# Generate SQL Query
def text_to_sql(user_query, schema_context):
    schema_context_str = "\n".join(schema_context)
    prompt_template = ChatPromptTemplate.from_messages([
            ("system",  "You are a database expert."),
            ("user", "For the database schema {schema_info}. Convert the following natural language request into SQL: {input_text}"),
            ("user", "Please reply only with query."),
            ("user", "If schema do not cover the question then reply - Out of syllabus !")
        ])
    chain = LLMChain(llm=llm, prompt=prompt_template)
    # Generate SQL using the chain
    sql_query = chain.run(input_text=user_query, schema_info=schema_context_str)
    return sql_query


def main():
    print("Welcome! Type 'exit' to quit.")
    schema_data = process_schema()
    # Map index to schema descriptions
    #user_query = "List all employee names who work in HR department and the project name they are working on"
    while True:
        user_input = input("You: ")
        
        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Generate a response
        schema_context = retrieve_relevant_schema(user_input, schema_data)
        generated_sql = text_to_sql(user_input, schema_context)
        print(f"Bot: {generated_sql}")

if __name__ == "__main__":
    main()
