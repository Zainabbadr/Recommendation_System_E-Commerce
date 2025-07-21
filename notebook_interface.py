"""
Jupyter Notebook interface for the E-Commerce Recommendation System.
Use this file in Jupyter notebooks for interactive exploration.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path.cwd() / "src"))

# Import main classes
from src.data.processor import DataProcessor
from src.models.recommendations import (
    CollaborativeFiltering, 
    ContentBasedFiltering, 
    CategoryClustering
)
from src.agents.crew_agents import RecommendationAgents
from src.utils.config import Config

# For Jupyter notebook compatibility
from IPython.display import display
import pandas as pd

class NotebookRecommendationSystem:
    """
    Jupyter Notebook-friendly version of the recommendation system.
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.data_processor = DataProcessor()
        self.cf_model = CollaborativeFiltering()
        self.cb_model = ContentBasedFiltering()
        self.categorizer = CategoryClustering()
        self.df = None
        
        print("🚀 E-Commerce Recommendation System initialized!")
        print("Available methods:")
        print("  - load_data(): Load and process the dataset")
        print("  - analyze(): Perform data analysis")
        print("  - collaborative_rec(user_id): Get collaborative filtering recommendations")
        print("  - content_rec(description): Get content-based recommendations")
    
    def load_data(self):
        """Load and process data with progress display."""
        print("📊 Loading dataset...")
        raw_df = self.data_processor.load_dataset()
        
        if raw_df is not None:
            print("🔧 Processing data...")
            self.df = self.data_processor.clean_data(raw_df)
            
            print("🏷️ Categorizing products...")
            descriptions = self.df['Description']
            desc_to_group = self.categorizer.fit_transform(descriptions)
            self.df['Description_Categorize'] = self.df['Description'].map(desc_to_group)
            
            print(f"✅ Data loaded and processed! Shape: {self.df.shape}")
            display(self.df.head())
            return self.df
        else:
            print("❌ Failed to load data")
            return None
    
    def analyze(self):
        """Display data analysis with nice formatting."""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return
        
        print("📈 Data Analysis")
        print("=" * 50)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Unique customers: {self.df['CustomerID'].nunique():,}")
        print(f"Unique products: {self.df['StockCode'].nunique():,}")
        print(f"Unique descriptions: {self.df['Description'].nunique():,}")
        
        print("\n🌍 District Distribution:")
        district_counts = self.df['District'].value_counts()
        display(district_counts.to_frame('Count'))
        
        print("\n🏳️ Top 10 Countries:")
        country_counts = self.df['Country'].value_counts().head(10)
        display(country_counts.to_frame('Count'))
    
    def collaborative_rec(self, user_id, top_n=10):
        """Get collaborative filtering recommendations."""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
        
        print(f"🤝 Getting collaborative recommendations for user {user_id}...")
        self.cf_model.fit(self.df)
        recommendations = self.cf_model.get_recommendations(user_id, self.df, top_n)
        
        if isinstance(recommendations, str):
            print(f"⚠️ {recommendations}")
            return None
        
        print(f"✅ Found {len(recommendations)} recommendations:")
        display(recommendations)
        return recommendations
    
    def content_rec(self, description, top_n=5):
        """Get content-based recommendations."""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
        
        print(f"🔍 Getting content-based recommendations for: '{description}'")
        self.cb_model.fit(self.df)
        recommendations = self.cb_model.get_recommendations(description, top_n)
        
        print(f"✅ Found {len(recommendations)} similar products:")
        display(recommendations)
        return recommendations
    
    def sample_users(self, n=10):
        """Display sample user IDs for testing."""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
        
        sample_users = self.df['CustomerID'].unique()[:n]
        print(f"📝 Sample user IDs for testing:")
        for i, user_id in enumerate(sample_users, 1):
            print(f"  {i}. User ID: {user_id}")
        return sample_users
    
    def sample_products(self, n=10):
        """Display sample product descriptions for testing."""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
        
        sample_products = self.df['Description'].unique()[:n]
        print(f"🛍️ Sample product descriptions for testing:")
        for i, desc in enumerate(sample_products, 1):
            print(f"  {i}. {desc}")
        return sample_products

# Initialize the system for notebook use
print("📓 Notebook interface ready!")
print("Create an instance: system = NotebookRecommendationSystem()")
print("Then start with: system.load_data()")
