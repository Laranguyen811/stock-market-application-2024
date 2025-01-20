import google.generativeai as genai
import os
import sqlite3
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

db_file = 'companies.db'
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
list_tables()
import os

def database_exists(db_name):
    """
    Check if the database file exists.
    """
    return os.path.isfile(db_name)

db_name = 'companies.db'
if database_exists(db_name):
    print(f"The database '{db_name}' exists.")
else:
    print(f"The database '{db_name}' does not exist.")

import sqlite3
