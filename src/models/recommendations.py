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
from sklearn.preprocessing import MinMaxScaler

class CollaborativeFiltering:
    """Collaborative filtering recommendation system."""

    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None

    def fit(self, df):
        """Fit the collaborative filtering model."""
        self.user_item_matrix = df.pivot_table(
            index="CustomerID", columns="StockCode", values="Quantity", aggfunc="mean"
        ).fillna(0)
        return self

    def get_recommendations(self, data, target_user_id, stock_codes, top_n=10):
        """Get recommendations for a target user."""
        try:
            if target_user_id not in data['CustomerID'].unique():
                return {
                    'user_id': target_user_id,
                    'status': f'CustomerID {target_user_id} not found',
                    'recommendations': []
                }

            invalid_stock_codes = [code for code in stock_codes if code not in data['StockCode'].unique()]
            if invalid_stock_codes:
                return {
                    'user_id': target_user_id,
                    'status': f'Stock codes not found: {invalid_stock_codes}',
                    'recommendations': []
                }

            user_item_matrix = data.pivot_table(
                index='CustomerID',
                columns='StockCode',
                values='Quantity',
                aggfunc='mean'
            ).fillna(0)

            # User’s current and added items
            current_items = set(user_item_matrix.loc[target_user_id][user_item_matrix.loc[target_user_id] > 0].index)
            updated_items = current_items.union(set(stock_codes))
            for item in updated_items:
                user_item_matrix.loc[target_user_id, item] = 1

            # Get district of target user
            target_district = data[data['CustomerID'] == target_user_id]['District'].mode().values[0]
            district_users = data[
                (data['District'] == target_district) &
                (data['CustomerID'] != target_user_id)
            ]['CustomerID'].unique()
            district_users = [uid for uid in district_users if uid in user_item_matrix.index]

            # Filter matrix and compute similarities
            filtered_matrix = user_item_matrix.loc[[target_user_id] + district_users]
            similarities = cosine_similarity(filtered_matrix)[0][1:]

            similar_users = sorted(zip(district_users, similarities), key=lambda x: x[1], reverse=True)[:10]

            # Score recommendations
            recommendations = defaultdict(lambda: {'score': 0, 'users': []})
            for similar_user, similarity in similar_users:
                similar_items = user_item_matrix.loc[similar_user]
                for item in similar_items[similar_items > 0].index:
                    if item not in updated_items:
                        recommendations[item]['score'] += similarity
                        recommendations[item]['users'].append(similar_user)

            if not recommendations:
                return {
                    'user_id': target_user_id,
                    'district': target_district,
                    'status': 'No new recommendations found',
                    'recommendations': []
                }

            # Format recommendation output
            result = []
            for item, rec_data in sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]:
                item_info = data[data['StockCode'] == item].iloc[0]
                result.append({
                    'stock_code': item,
                    'description': item_info['Description'],
                    'price': round(float(item_info['UnitPrice']), 2),
                    'recommended_by': rec_data['users']
                })

            # Get user items info (original + added)
            user_items_info = data[data['StockCode'].isin(updated_items)]
            user_items_summary = user_items_info.groupby('StockCode').agg({
                'Description': 'first',
                'UnitPrice': 'mean'
            }).reset_index()

            user_items_list = user_items_summary.to_dict(orient='records')
            for item in user_items_list:
                item['UnitPrice'] = round(float(item['UnitPrice']), 2)

            return {
                'user_id': target_user_id,
                'district': target_district,
                'status': 'success',
                'user_items': user_items_list,
                'recommendations': result
            }

        except Exception as e:
            return {
                'user_id': target_user_id,
                'status': f'Error: {str(e)}',
                'recommendations': []
            }




class ContentBasedFiltering:
    """Content-based filtering recommendation system."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.data = None

    def fit(self, df):
        """Fit the content-based filtering model."""
        self.data = df
        self.tfidf_matrix = self.vectorizer.fit_transform(df["Description"])
        return self

    def get_recommendations(self, stock_codes, data, target_user_id=None, top_n=5):
        """Get recommendations based on input description."""
        # Preprocess input description
        try:
            # Vectorize descriptions once
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(data['Description'])

            all_seen_descriptions = set()
            all_results = []
            user_items_info = []

            for stock in stock_codes:
                indices = data.index[data['StockCode'] == stock].tolist()
                if not indices:
                    all_results.append({
                        "stock_code": stock,
                        "status": f"Stock code '{stock}' not found",
                        "recommendations": []
                    })
                    continue

                input_idx = indices[0]
                input_vector = tfidf_matrix[input_idx]
                similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

                input_description = data.at[input_idx, "Description"]
                input_unit_price = float(data.at[input_idx, "UnitPrice"])  # convert here
                user_items_info.append({
                    "stock_code": stock,
                    "description": input_description,
                    "unit_price": input_unit_price
                })

                seen_descriptions = set(all_seen_descriptions)
                seen_descriptions.add(input_description)

                sorted_indices = similarity_scores.argsort()[::-1]
                recommendations = []

                for idx in sorted_indices:
                    if idx == input_idx:
                        continue
                    desc = data.at[idx, "Description"]
                    if desc not in seen_descriptions:
                        recommendations.append({
                            "stock_code": data.at[idx, "StockCode"],
                            "description": desc,
                            "unit_price": float(data.at[idx, "UnitPrice"])  # convert here too
                        })
                        seen_descriptions.add(desc)
                        all_seen_descriptions.add(desc)

                    if len(recommendations) >= top_n:
                        break

                status = "success" if recommendations else "No similar recommendations found"

                all_results.append({
                    "stock_code": stock,
                    "status": status,
                    "recommendations": recommendations
                })

            return {
                "target_user_id": target_user_id,
                "target_user_items": user_items_info,
                "results": all_results
            }

        except Exception as e:
            return {
                "target_user_id": target_user_id,
                "status": f"Error: {str(e)}",
                "results": []
            }

class CategoryClustering:
    """Clustering for product categorization."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.similarity_matrix = None

    def fit_transform(self, descriptions, similarity_threshold=0.60):
        """Cluster descriptions based on similarity."""
        # Clean descriptions
        clean_descriptions = [
            desc
            for desc in descriptions
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
                    if (
                        group_labels[j] == -1
                        and self.similarity_matrix[i][j] >= similarity_threshold
                    ):
                        group_labels[j] = current_group
                current_group += 1

        return dict(zip(clean_descriptions, group_labels))


@tool("retail_recommendation_engine")
def collaborative_filtering_recommendations(
    df_json: str, target_user_id: int, top_n: int = 10
):
    """
    Generate product recommendations using collaborative filtering.
    """
    import json
    from io import StringIO

    # Convert JSON string back to DataFrame
    df = pd.read_json(StringIO(df_json))

    # Run the original logic
    cf_model = CollaborativeFiltering()
    cf_model.fit(df)
    recommendations = cf_model.get_recommendations(target_user_id, df, top_n)

    # Return as string for CrewAI compatibility
    if isinstance(recommendations, pd.DataFrame):
        return f"Collaborative Filtering Recommendations:\n{recommendations.to_string(index=False)}"
    else:
        return f"Collaborative Filtering Result: {recommendations}"
    

def weighted_hybrid_recommendations(input_stock_code, target_user_id, data, alpha=0.6, top_n=5):
    try:
        # Vectorize descriptions
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(data['Description'])

        # Locate input stock code
        indices = data.index[data['StockCode'] == input_stock_code].tolist()
        if not indices:
            return {"Top Recommendations": []}

        input_idx = indices[0]
        input_vector = tfidf_matrix[input_idx]

        # Content-based scores
        content_sim_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

        # Collaborative filtering setup
        user_item_matrix = data.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            aggfunc='mean'
        ).fillna(0)

        if target_user_id not in user_item_matrix.index:
            return {"Top Recommendations": []}

        target_district = data[data['CustomerID'] == target_user_id]['District'].mode().values[0]
        district_users = data[
            (data['District'] == target_district) &
            (data['CustomerID'] != target_user_id)
        ]['CustomerID'].unique()
        district_users = [uid for uid in district_users if uid in user_item_matrix.index]

        filtered_matrix = user_item_matrix.loc[[target_user_id] + district_users]
        similarities = cosine_similarity(filtered_matrix)[0][1:]

        similar_users = sorted(zip(district_users, similarities), key=lambda x: x[1], reverse=True)[:10]

        # Collaborative filtering scores
        collab_scores = np.zeros(len(data))
        for similar_user, sim in similar_users:
            similar_items = user_item_matrix.loc[similar_user]
            for stock in similar_items[similar_items > 0].index:
                idxs = data.index[data['StockCode'] == stock].tolist()
                for idx in idxs:
                    collab_scores[idx] += sim

        # Normalize both scores
        scaler = MinMaxScaler()
        content_norm = scaler.fit_transform(content_sim_scores.reshape(-1, 1)).flatten() if len(set(content_sim_scores)) > 1 else content_sim_scores
        collab_norm = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten() if len(set(collab_scores)) > 1 else collab_scores

        # Combine scores
        final_scores = alpha * content_norm + (1 - alpha) * collab_norm
        top_indices = final_scores.argsort()[::-1]

        input_desc = data.at[input_idx, 'Description']
        seen_descriptions = {input_desc.lower()}
        recommendations = []

        for idx in top_indices:
            if idx == input_idx or final_scores[idx] <= 0:
                continue

            desc = data.at[idx, 'Description']
            if desc.lower() in seen_descriptions:
                continue

            recommendations.append({
                "Stock Code": data.at[idx, 'StockCode'],
                "Description": desc,
                "Unit Price": round(float(data.at[idx, 'UnitPrice']), 2)
            })

            seen_descriptions.add(desc.lower())

            if len(recommendations) >= top_n:
                break

        return {"Top Recommendations": recommendations}

    except Exception as e:
        return {"Top Recommendations": [], "Error": str(e)}

