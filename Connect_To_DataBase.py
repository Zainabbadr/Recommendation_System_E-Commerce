from sqlalchemy import create_engine, text
import pandas as pd
import pymssql
from dotenv import load_dotenv
import os

load_dotenv()
server = os.getenv('server')
user = os.getenv('user')
database = os.getenv('database')
password = os.getenv('password')


def connect_to_database():
    try:
        conn = pymssql.connect(server=server, user=user, password=password, database=database)
        engine = create_engine('mssql+pymssql://', creator=lambda: conn)
        return engine
    except Exception as e:
        print("❌ Database connection failed:", e)
        return None


def execute_query(engine, query):
    try:
        with engine.connect() as connection:
            result = pd.read_sql(query, connection)
            return result
    except Exception as e:
        print("❌ Failed to execute query:", e)
        return None
