import google.generativeai as genai
import os
import sqlite3
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

db_file = 'stock_companies.db'
db_conn = sqlite3.connect(db_file)

def list_tables() -> list[str]:
    '''
    Retrieve the names of all tables in the database.
    '''
    print(' - DB call: list_tables')
    cursor = db_conn.cursor()

    # Fetch the table names
    cursor.execute('''
    SELECT name FROM sqlite_master WHERE type='table'
    ''')
    table = cursor.fetchall()
    cursor.close()
    return[t[0] for t in table]
table_names = list_tables() # When calling list_tables() function, ensure that its returned value is printed or used.
print(table_names)

def describe_table(table_name: str) -> list[tuple[str,str]]:
    '''
    Look up the table schema.
    Returns:
        List of columns, where each entry is a tuple of (column, type).
    '''
    print(' - DB call: describe_table')
    cursor = db_conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})") # PRAGMA is a special SQLite command to modify operations or obtain metadata (querying the internal structures of SQL data).
    schema = cursor.fetchall()
    return [(col[1], col[2]) for col in schema]
describe_companies = describe_table('stock_companies')
print(describe_companies)

def execute_query(sql: str) -> list[list[str]]:
    '''
    Execute a SELECT statement, returning the results.
    '''
    print(' - DB call: execute_query')
    cursor = db_conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()
select_stock = execute_query(" select * from stock_companies")
print(select_stock)

db_tools = [list_tables,describe_table,execute_query]
instruction = """ You are a helpful data analyst that can interact with a SQL database for 
a stock market platform. You will take the users questions and turn them into SQL queries using the tools
available. Once you have the information you need, you will answer the user's question using the data returned.
Use list_tables to see what tables are available, describe_table to understand the schema of the table, and execute_query to
issue an SQL SELECT query.
"""