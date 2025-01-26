from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, URL
from langchain_aws import ChatBedrock as BedrockChat
from pyathena.sqlalchemy.rest import AthenaRestDialect
import os

class CustomAthenaRestDialect(AthenaRestDialect):
    def import_dbapi(self):
        import pyathena
        return pyathena

# DB Connection details
connathena = "athena.us-west-2.amazonaws.com"
portathena = '443'
schemaathena = 'mycur'
s3stagingathena = 's3://cur-data-test01/athena-query-result/'
wkgrpathena = 'primary'
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
url = URL.create("awsathena+rest", query={"s3_staging_dir": s3stagingathena, "work_group": wkgrpathena})
engine_athena = create_engine(url, dialect=CustomAthenaRestDialect(), echo=False)
db = SQLDatabase(engine_athena)

# Setup LangChain and LLM (Bedrock)
model_kwargs = {"temperature": 0, "top_k": 250, "top_p": 1, "stop_sequences": ["\n\nHuman:"]}
llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs=model_kwargs)

# Query Template
QUERY = """
Create a syntactically correct Athena query for AWS Cost and Usage Report to run on the my_c_u_r table in the mycur database based on the question, then execute the query and return the results.
{question}
"""

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

def lambda_handler(event, context):
    user_input = event.get('user_input')
    
    if not user_input:
        return {"statusCode": 400, "body": "User input is required."}
    
    # Format and process the query
    question = QUERY.format(question=user_input)
    result = db_chain.invoke(question)
    query = result["result"].split("SQLQuery:")[1].strip()
    rows = db.run(query)
    
    response = {
        "SQLQuery": query,
        "SQLResult": rows
    }
    
    return {
        "statusCode": 200,
        "body": response
    }

# Just for testing locally
if __name__ == "__main__":
    # Example Test Event
    event = {"user_input": "What is my AWS cost for the last month?"}
    context = None
    response = lambda_handler(event, context)
    print(response)
