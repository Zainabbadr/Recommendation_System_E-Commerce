import sys
import os
from pathlib import Path
# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.crew_agents import RecommendationAgents

def main():
    # Example customer ID
    target_user_id = 17850
    
    # Example number of recommendations to return
    top_n = 5
    
    # Optional list of stock codes to filter recommendations
    # Set to None to get recommendations across all products
    stock_codes = ["85123A","71053","84406B"]  # Example: ["85123A", "71053", "84406B"]
    
    print(f"🔍 Getting recommendations for customer {target_user_id}...")
    
    # Initialize the recommendation agents
    agents = RecommendationAgents()
    
    # Run recommendations with the simplified inputs
    results = agents.run_recommendations(
        target_user_id=target_user_id,
        top_n=top_n,
        stock_codes=stock_codes
    )
    
    print("\n=== Recommendation Results ===")
    print(results)

if __name__ == "__main__":
    main()