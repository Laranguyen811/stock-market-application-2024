import os
from py2neo import Graph
from py2neo.errors import ServiceUnavailable

# Load credentials from environment variables
NEO4J_URI = os.getenv('NEO4J_URI','http://localhost:7474')  # Obtaining neo4j uri
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME','neo4j')  # Obtaining neo4j username
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD','root')  # Obtaining neo4j password

try:  # Attempting a block of code before throwing an exception
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME,NEO4J_PASSWORD))  #Connecting to the Neo4j database using given credentials

    # Deleting all nodes and relationships
    graph.delete_all()
    print("All nodes and relationships have been deleted.")
except ServiceUnavailable as e:  # Throwing an exception when service is unavailable as variable e
    print(f"Failed to connect to the database: {e}")
except Exception as e:  # Throwing an exception error as variable e
    print(f"An error occurred: {e}")

