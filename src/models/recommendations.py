"""
Recommendation system models and algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from langchain_core.tools import tool


class CollaborativeFiltering:
    """Collaborative filtering recommendation system."""
    
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
    
    def fit(self, df):
        """Fit the collaborative filtering model."""
        self.user_item_matrix = df.pivot_table(
            index='CustomerID', 
            columns='StockCode', 
            values='Quantity', 
            aggfunc='mean'
        ).fillna(0)
        return self
    
    def get_recommendations(self, target_user_id, df, top_n=10):
        """Get recommendations for a target user."""
        if target_user_id not in self.user_item_matrix.index:
            return f"CustomerID {target_user_id} not found."

        # Get target user's district
        target_district = df[df['CustomerID'] == target_user_id]['District'].mode().values[0]

        # Filter to only customers in the same district
        same_district_users = df[df['District'] == target_district]['CustomerID'].unique()
        same_district_users = [
            uid for uid in same_district_users 
            if uid in self.user_item_matrix.index and uid != target_user_id
        ]

        # Create new matrix only for these users
        filtered_user_item_matrix = self.user_item_matrix.loc[
            [target_user_id] + same_district_users
        ]

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(filtered_user_item_matrix)
        target_index = filtered_user_item_matrix.index.get_loc(target_user_id)
        similarities = similarity_matrix[target_index]

        # Get top 10 similar users (excluding self)
        similar_users = [
            (uid, similarities[i]) 
            for i, uid in enumerate(filtered_user_item_matrix.index) 
            if uid != target_user_id
        ]
        top_similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[:10]
        top_user_ids = [uid for uid, _ in top_similar_users]

        # Recommendation logic
        target_user_row = self.user_item_matrix.loc[target_user_id]
        item_quantity_map = defaultdict(float)
        item_user_map = defaultdict(list)

        for uid in top_user_ids:
            similar_user_row = self.user_item_matrix.loc[uid]
            recommended_mask = (similar_user_row > 0) & (target_user_row == 0)
            recommended_items = self.user_item_matrix.columns[recommended_mask]

            for item in recommended_items:
                item_quantity_map[item] += similar_user_row[item]
                item_user_map[item].append(uid)

        # Build result
        if not item_quantity_map:
            return pd.DataFrame(columns=[
                'StockCode', 'Description', 'TotalQuantity', 'Users', 'District'
            ])

        rec_df = pd.DataFrame([
            {
                'StockCode': item,
                'TotalQuantity': item_quantity_map[item],
                'Users': item_user_map[item],
                'District': target_district
            }
            for item in item_quantity_map
        ])

        rec_df = rec_df.sort_values(by='TotalQuantity', ascending=False).head(top_n)
        rec_df = rec_df.merge(
            df[['StockCode', 'Description']].drop_duplicates(), 
            on='StockCode', 
            how='left'
        )

        return rec_df[['StockCode', 'Description', 'TotalQuantity', 'Users', 'District']]


class ContentBasedFiltering:
    """Content-based filtering recommendation system."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.data = None
    
    def fit(self, df):
        """Fit the content-based filtering model."""
        self.data = df
        self.tfidf_matrix = self.vectorizer.fit_transform(df['Description'])
        return self
    
    def get_recommendations(self, input_description, top_n=5):
        """Get recommendations based on input description."""
        # Preprocess input description
        clean_input = input_description.lower()
        clean_input = ''.join(char for char in clean_input if char.isalnum() or char.isspace())

        # Vectorize input
        input_vec = self.vectorizer.transform([clean_input])

        # Compute similarity between input and all items
        sim_scores = linear_kernel(input_vec, self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[::-1]

        # Filter out exact duplicates of input description
        seen = set()
        recommendations = []
        for i in sim_indices:
            desc = self.data.iloc[i]['Description']
            if desc != clean_input and desc not in seen:
                seen.add(desc)
                recommendations.append(i)
            if len(recommendations) == top_n:
                break

        return self.data.iloc[recommendations][['Description']]


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


@tool("retail_recommendation_engine")
def collaborative_filtering_recommendations(df, target_user_id, top_n=10):
    """
    Generate product recommendations using collaborative filtering.
    """
    cf_model = CollaborativeFiltering()
    cf_model.fit(df)
    return cf_model.get_recommendations(target_user_id, df, top_n)
