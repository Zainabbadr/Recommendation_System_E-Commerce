"""
Data processing utilities for the e-commerce recommendation system.
"""

import os
import pandas as pd
import kagglehub
from langchain_core.tools import tool


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

            # North America
            'USA': 'North America',
            'Canada': 'North America',

            # Other
            'Australia': 'Oceania',
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
    
    def clean_data(self, df):
        """Clean and preprocess the dataframe."""
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
        
        # Add District column based on Country
        df['District'] = df['Country'].map(self.country_to_district)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        return df


@tool("retail_data_preparation")
def prepare_data(df_json: str):
    """
    Prepare the retail transaction data by:
    - Dropping rows with missing CustomerID or Description
    - Converting CustomerID to integer
    - Cleaning column names
    - Removing non-numeric StockCodes
    - Removing BANK CHARGES
    - Adding District column based on Country
    - Removing duplicates
    """
    import json
    from io import StringIO
    
    # Convert JSON string back to DataFrame
    df = pd.read_json(StringIO(df_json))
    
    # Run the original logic
    processor = DataProcessor()
    cleaned_df = processor.clean_data(df)
    
    # Return summary as string for CrewAI compatibility
    return f"Data preparation completed:\n- Original shape: {df.shape}\n- Cleaned shape: {cleaned_df.shape}\n- Districts added: {cleaned_df['District'].value_counts().to_dict()}"
