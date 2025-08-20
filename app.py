from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.llms import HuggingFaceHub


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    pw = quote_plus(password)                 # <- handles special chars
    uri = f"mysql+mysqlconnector://{user}:{pw}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(uri)


load_dotenv()
api_key = "GROQ_API_KEY"



def get_sql_chain(db: SQLDatabase, row_limit_default: int = 50):
    template = """
You are a senior MySQL analyst. Write a single MySQL SELECT query only.
No prose, no code fences, no backticks.

Rules:
- Use ONLY tables/columns from the schema.
- Prefer explicit JOIN ... ON ... using keys when joining tables.
- If user asks for "top N", use ORDER BY a meaningful metric and LIMIT N.
- If user did not specify a limit, append LIMIT {row_limit} at the end.
- Dates are MySQL dialect. Avoid functions not supported by MySQL.

<SCHEMA>
{schema}
</SCHEMA>

User question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

    def get_schema(_input):
        # Optionally: return db.get_table_info(table_names=["orders","customers",...])
        return db.get_table_info()

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema, row_limit=lambda _: row_limit_default)
        | prompt
        | llm
        | StrOutputParser()
    )
    return sql_chain
def run_query(db: SQLDatabase, question: str, sql_chain):
    sql = sql_chain.invoke({"question": question})
    result = db.run(sql)                      # returns rows (list of tuples) or a string
    return sql, result

def get_nl_chain():
        template = """
        You are a helpful data analyst.
        User asked: {question}
        Rows returned: {result}

        Summarize the answer in clear natural language (short).
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm =ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

        return prompt | llm | StrOutputParser()
    
st.set_page_config(page_title="Chat with my SQL", page_icon="💬")
st.title("Chat with my SQL")

# One-time init
if "db" not in st.session_state:
    st.session_state.db = init_database(
        user="root",
        password="Umamaheswari@1981!",
        host="localhost",
        port="3306",
        database="sql_practice",
    )
    st.session_state.sql_chain = get_sql_chain(st.session_state.db)
    st.session_state.nl_chain = get_nl_chain()

question = st.text_input("Ask your question about the database")

if question:
    with st.spinner("Generating SQL and fetching results…"):
        try:
            sql, result = run_query(st.session_state.db, question, st.session_state.sql_chain)
            st.subheader("Generated SQL")
            st.code(sql, language="sql")

            st.subheader("Query Results")
            st.write(result)

            with st.spinner("Summarizing…"):
                answer = st.session_state.nl_chain.invoke({"question": question, "result": result})
            st.subheader("Query Details")
            st.write(answer)
        except Exception as e:
            st.error(f"Query failed: {e}")    
print(api_key[:5])            
        




