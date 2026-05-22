import os 
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
    )) 

