"""
Main application file for the E-Commerce Recommendation System.

This is a comprehensive recommendation system that includes:
- Data processing and cleaning
- Collaborative filtering
- Content-based filtering  
- Product categorization using sentence transformers
- CrewAI agents for automated recommendation workflows
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.processor import DataProcessor
from src.models.recommendations import (
    CollaborativeFiltering, 
    ContentBasedFiltering
)
# Try to import agents, with fallback for dependency issues
try:
    from src.agents.crew_agents import RecommendationAgents
    AGENTS_AVAILABLE = True
    print("✅ CrewAI agents loaded successfully")
except ImportError as e:
    print(f"⚠️ CrewAI agents not available due to dependency issue:")
    print(f"   Error: {str(e)[:100]}...")
    print("   Running without AI agents (core functionality still works)")
    RecommendationAgents = None
    AGENTS_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Unexpected error loading CrewAI agents:")
    print(f"   Error: {str(e)[:100]}...")
    print("   Running without AI agents (core functionality still works)")
    RecommendationAgents = None
    AGENTS_AVAILABLE = False
from src.utils.config import Config


class ECommerceRecommendationSystem:
    """Main class for the e-commerce recommendation system."""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.data_processor = DataProcessor()
        self.cf_model = CollaborativeFiltering()
        self.cb_model = ContentBasedFiltering()
        
        # Only create agents if available
        if AGENTS_AVAILABLE and RecommendationAgents:
            self.agents = RecommendationAgents(
                api_key=self.config.api.google_api_key
            )
        else:
            self.agents = None
            print("⚠️ AI agents disabled due to dependency issues")
        
        self.df = None
    
    def load_and_process_data(self):
        """Load and process the e-commerce dataset."""
        print("Loading dataset...")
        raw_df = self.data_processor.load_dataset()
        
        if raw_df is not None:
            print("Processing data...")
            self.df = self.data_processor.clean_data(raw_df)
            print(f"Processed data shape: {self.df.shape}")
            print("✅ Data processing completed")
            
            return self.df
        return None
    
    def get_collaborative_recommendations(self, target_user_id, top_n=None):
        """Get collaborative filtering recommendations."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        top_n = top_n or self.config.model.collaborative_filtering_top_n
        
        print(f"Getting collaborative filtering recommendations for user {target_user_id}...")
        self.cf_model.fit(self.df)
        recommendations = self.cf_model.get_recommendations(target_user_id, self.df, top_n)
        return recommendations
    
    def get_content_based_recommendations(self, description, top_n=None):
        """Get content-based recommendations."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        top_n = top_n or self.config.model.content_based_top_n
        
        print(f"Getting content-based recommendations for: {description}")
        self.cb_model.fit(self.df)
        recommendations = self.cb_model.get_recommendations(description, top_n)
        return recommendations
    
    def run_crew_ai_workflow(self, target_user_id):
        """Run the CrewAI agent workflow."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        if not AGENTS_AVAILABLE or self.agents is None:
            print("❌ CrewAI agents not available. Skipping AI workflow.")
            print("   You can still use collaborative_recommendations() and content_based_recommendations()")
            return None
        
        print("Running CrewAI workflow...")
        results = self.agents.run_recommendations(self.df, target_user_id)
        return results
    
    def analyze_data(self):
        """Perform basic data analysis."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        print("\n=== Data Analysis ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Unique customers: {self.df['CustomerID'].nunique()}")
        print(f"Unique products: {self.df['StockCode'].nunique()}")
        print(f"Unique descriptions: {self.df['Description'].nunique()}")
        print("\nDistrict distribution:")
        print(self.df['District'].value_counts())
        print("\nCountry distribution (top 10):")
        print(self.df['Country'].value_counts().head(10))


def main():
    """Main function to demonstrate the recommendation system."""
    # Initialize the system
    system = ECommerceRecommendationSystem()
    
    try:
        # Load and process data (fast processing without categorization)
        df = system.load_and_process_data()
        
        if df is not None:
            # Analyze data
            system.analyze_data()
            
            # Example: Get collaborative filtering recommendations
            target_user_id = 17850
            cf_recommendations = system.get_collaborative_recommendations(target_user_id)
            print(f"\n=== Collaborative Filtering Recommendations for User {target_user_id} ===")
            print(cf_recommendations)
            
            # Example: Get content-based recommendations
            description = "white hanging heart t-light holder"
            cb_recommendations = system.get_content_based_recommendations(description)
            print(f"\n=== Content-Based Recommendations for '{description}' ===")
            print(cb_recommendations)
            
            # Example: Run CrewAI workflow
            if AGENTS_AVAILABLE and system.agents:
                try:
                    crew_results = system.run_crew_ai_workflow(target_user_id)
                    if crew_results:
                        print("\n=== CrewAI Workflow Results ===")
                        print(crew_results)
                except Exception as e:
                    print(f"CrewAI workflow failed: {e}")
            else:
                print("\n⚠️ CrewAI workflow skipped (agents not available)")
                print("✅ Core recommendation features are working!")
                print("   - Collaborative filtering: ✅")
                print("   - Content-based filtering: ✅")  
                print("   - Data processing: ✅")
    
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
