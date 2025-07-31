"""
Data processing utilities for the e-commerce recommendation system.
"""

import os
import pandas as pd
import kagglehub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Add Django imports for SQLite functionality
try:
    from django.db import connection
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    print("⚠️ Django not available - SQLite loading will not work")

class DataProcessor:
    """Class for handling e-commerce data processing operations."""
    
    def __init__(self):
        self.country_to_district = {
            # Europe
            'United Kingdom': 'Europe',
            'France': 'Europe',
            'Germany': 'Europe',
            'Spain': 'Europe',
            'Portugal': 'Europe',
            'Italy': 'Europe',
            'Netherlands': 'Europe',
            'Switzerland': 'Europe',
            'Belgium': 'Europe',
            'Austria': 'Europe',
            'Sweden': 'Europe',
            'Finland': 'Europe',
            'Denmark': 'Europe',
            'Norway': 'Europe',
            'Lithuania': 'Europe',
            'Greece': 'Europe',
            'Poland': 'Europe',
            'Cyprus': 'Europe',
            'Malta': 'Europe',
            'Iceland': 'Europe',
            'Channel Islands': 'Europe',
            'European Community': 'Europe',
            'EIRE': 'Europe',
            'Czech Republic': 'Europe',

            # Middle East
            'Saudi Arabia': 'Middle East',
            'Lebanon': 'Middle East',
            'United Arab Emirates': 'Middle East',
            'Israel': 'Middle East',
            'Bahrain': 'Middle East',

            # Asia-Pacific
            'Japan': 'Asia-Pacific',
            'Singapore': 'Asia-Pacific',
            'Australia': 'Asia-Pacific',

            # North America
            'USA': 'North America',
            'Canada': 'North America',

            # South America
            'Brazil': 'South America',
            'RSA': 'Africa',
            'Unspecified': 'Unknown'
        }
    
    def load_dataset(self):
        """Load the e-commerce dataset from Kaggle."""
        path = kagglehub.dataset_download("carrie1/ecommerce-data")
        print(f"Path to dataset files: {path}")
        
        csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
        print(f"CSV files: {csv_files}")
        
        if csv_files:
            df = pd.read_csv(
                os.path.join(path, csv_files[0]), 
                encoding='ISO-8859-1'
            )
            print(f"Loaded DataFrame shape: {df.shape}")
            return df
        else:
            print("No CSV file found in the dataset.")
            return None

    def load_data_from_sqlite(self):
        """Load data from SQLite database instead of CSV files."""
        if not DJANGO_AVAILABLE:
            print("❌ Django not available - cannot load from SQLite")
            return None
            
        try:
            print("🔧 Loading data from SQLite database...")
            
            # SQL query to join all tables and get the data in the format expected by the recommendation algorithms
            query = """
            SELECT 
                ft.InvoiceNo,
                ft.CustomerID_id as CustomerID,
                ft.StockCode_id as StockCode,
                ft.Quantity,
                ft.UnitPrice,
                ft.TotalPrice,
                ft.InvoiceDate,
                dp.Description,
                dp.Description_Categorize,
                dc.Country,
                dc.District,
                dc.Customer_TotalSpending,
                dc.Segment
            FROM recommendations_fact_transactions ft
            LEFT JOIN recommendations_dim_products dp ON ft.StockCode_id = dp.StockCode
            LEFT JOIN recommendations_dim_customers dc ON ft.CustomerID_id = dc.CustomerID
            ORDER BY ft.InvoiceDate
            """
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, connection)
            
            # Convert InvoiceDate to datetime
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            print(f"✅ Loaded {len(df)} rows from SQLite database")
            print(f"📊 Columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"📊 Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data from SQLite: {e}")
            return None
    
    def clean_data(self, df=None):
        """Clean and preprocess the dataframe. If no dataframe is provided, load it first."""
        # If no dataframe is provided, try to load from SQLite first, then fallback to CSV
        if df is None:
            df = self.load_data_from_sqlite()
            if df is None:
                print("⚠️ SQLite loading failed, falling back to CSV...")
                df = self.load_dataset()
                if df is None:
                    return None
                
        df = pd.DataFrame(df)
        # Drop rows with missing CustomerID and Description
        df = df.dropna(subset=['CustomerID', 'Description']).copy()
        
        # Convert CustomerID to integer
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Remove rows where StockCode contains only letters
        df = df[~df['StockCode'].str.match(r'^[A-Za-z]+$', na=False)]
        
        # Remove BANK CHARGES
        df = df[df['StockCode'] != 'BANK CHARGES']
        
        # Add District column based on Country if it doesn't exist
        # if 'District' not in df.columns:
        #     df['District'] = df['Country'].map(self.country_to_district)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        return df

    def cluster_categories(self, descriptions, similarity_threshold=0.60, model_name='all-MiniLM-L6-v2'):
        """Cluster product descriptions based on similarity."""        
        # Initialize model
        model = SentenceTransformer(model_name)
        
        # Clean descriptions
        clean_descriptions = [
            desc for desc in descriptions 
            if isinstance(desc, str) and len(desc.strip()) > 5
        ]
        
        # Generate embeddings
        embeddings = model.encode(clean_descriptions, show_progress_bar=True)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Clustering logic
        n = len(clean_descriptions)
        group_labels = [-1] * n
        current_group = 0

        for i in range(n):
            if group_labels[i] == -1:
                group_labels[i] = current_group
                for j in range(i + 1, n):
                    if (group_labels[j] == -1 and 
                        similarity_matrix[i][j] >= similarity_threshold):
                        group_labels[j] = current_group
                current_group += 1

        return dict(zip(clean_descriptions, group_labels))
    
# This entire section should be deleted:

class CategoryClustering:
    """Clustering for product categorization."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.similarity_matrix = None
    
    def fit_transform(self, descriptions, similarity_threshold=0.60):
        """Cluster descriptions based on similarity."""
        # Clean descriptions
        clean_descriptions = [
            desc for desc in descriptions 
            if isinstance(desc, str) and len(desc.strip()) > 5
        ]
        
        # Generate embeddings
        self.embeddings = self.model.encode(clean_descriptions, show_progress_bar=True)
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        # Clustering logic
        n = len(clean_descriptions)
        group_labels = [-1] * n
        current_group = 0

        for i in range(n):
            if group_labels[i] == -1:
                group_labels[i] = current_group
                for j in range(i + 1, n):
                    if (group_labels[j] == -1 and 
                        self.similarity_matrix[i][j] >= similarity_threshold):
                        group_labels[j] = current_group
                current_group += 1

        return dict(zip(clean_descriptions, group_labels))