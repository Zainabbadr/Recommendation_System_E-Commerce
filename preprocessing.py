import pandas as pd
import kagglehub
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime as dt


def load_and_clean_data():
    path = kagglehub.dataset_download("carrie1/ecommerce-data")
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    df = pd.read_csv(os.path.join(path, csv_files[0]), encoding='ISO-8859-1')

    df.columns = df.columns.str.strip()
    df = df[df['Quantity'] > 0]
    df = df.dropna(subset=['CustomerID', 'Description']).copy()
    df = df[~df['StockCode'].str.match(r'^[A-Za-z]+$', na=False)]
    df = df[df['StockCode'] != 'BANK CHARGES']
    df.drop_duplicates(inplace=True)

    country_to_district = {
        'United Kingdom': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Spain': 'Europe',
        'Portugal': 'Europe', 'Italy': 'Europe', 'Netherlands': 'Europe', 'Switzerland': 'Europe',
        'Belgium': 'Europe', 'Austria': 'Europe', 'Sweden': 'Europe', 'Finland': 'Europe',
        'Denmark': 'Europe', 'Norway': 'Europe', 'Lithuania': 'Europe', 'Greece': 'Europe',
        'Poland': 'Europe', 'Cyprus': 'Europe', 'Malta': 'Europe', 'Iceland': 'Europe',
        'Channel Islands': 'Europe', 'European Community': 'Europe',
        'Saudi Arabia': 'Middle East', 'Lebanon': 'Middle East', 'United Arab Emirates': 'Middle East',
        'Israel': 'Middle East', 'Bahrain': 'Middle East',
        'Japan': 'Asia-Pacific', 'Singapore': 'Asia-Pacific',
        'USA': 'North America', 'Canada': 'North America',
        'Australia': 'Oceania', 'EIRE': 'Europe', 'Brazil': 'South America',
        'RSA': 'Africa', 'Czech Republic': 'Europe', 'Unspecified': 'Unknown'
    }
    df['District'] = df['Country'].map(country_to_district)
    return df


def categorize_descriptions(df):
    descriptions = df['Description'].dropna().unique()
    clean_descriptions = [desc for desc in descriptions if isinstance(desc, str) and len(desc.strip()) > 5]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(clean_descriptions, show_progress_bar=True)
    cos_sim_matrix = cosine_similarity(embeddings)
    
    similarity_threshold = 0.60
    n = len(clean_descriptions)
    group_labels = [-1] * n
    current_group = 0

    for i in range(n):
        if group_labels[i] == -1:
            group_labels[i] = current_group
            for j in range(i + 1, n):
                if group_labels[j] == -1 and cos_sim_matrix[i][j] >= similarity_threshold:
                    group_labels[j] = current_group
            current_group += 1

    desc_to_group = dict(zip(clean_descriptions, group_labels))
    df['Description_Categorize'] = df['Description'].map(desc_to_group)
    return df


def calculate_spending(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    customer_total_spending = df.groupby('CustomerID')['TotalPrice'].sum()
    df['Customer_TotalSpending'] = df['CustomerID'].map(customer_total_spending)
    return df


def perform_rfm_segmentation(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    today_date = dt.datetime(2011, 12, 11)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda date : (today_date - date.max()).days,
        "InvoiceNo": lambda num: num.nunique(),
        "TotalPrice": lambda TotalPrice: TotalPrice.sum()
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm = rfm[rfm["Monetary"] > 0]

    rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating', r'[1-2][3-4]': 'at_Risk', r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep', r'33': 'need_attention', r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising', r'51': 'new_customers', r'[4-5][2-3]': 'potential_loyalists', r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm.reset_index()
    return df.merge(rfm[['CustomerID', 'segment']], on='CustomerID', how='right'), rfm


def preprocess_all():
    df = load_and_clean_data()
    df = categorize_descriptions(df)
    df = calculate_spending(df)
    df, rfm = perform_rfm_segmentation(df)
    return df, rfm
