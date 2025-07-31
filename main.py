from src.agents.crew_agents import RecommendationAgents
from src.data.processor import DataProcessor
import os
import sys
import django
from pathlib import Path
import pandas as pd

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_frontend.settings')
django.setup()

def main():
    # Load and process data
    print("📊 Loading and processing data...")
    processor = DataProcessor()

    # Load dataset from SQLite instead of CSV
    df = processor.load_data_from_sqlite()
    if df is None:
        print("❌ Failed to load dataset from SQLite")
        print("⚠️ Attempting to load from CSV as fallback...")
        df = processor.load_dataset()
        if df is None:
            print("❌ Failed to load dataset from both SQLite and CSV")
            return

    # Clean data
    df_clean = processor.clean_data(df)
    print(f"✅ Data loaded and cleaned: {len(df_clean)} rows")
    print(df_clean.head())

#     # Example customer ID - validate it exists
#     target_user_id = 17850

#     # Check if customer exists
#     if target_user_id not in df_clean["CustomerID"].unique():
#         print(f"❌ Customer {target_user_id} not found in dataset")
#         available_customers = df_clean["CustomerID"].unique()[:10]
#         print(f"Available customer IDs (first 10): {available_customers}")
#         return

#     # Example number of recommendations to return
#     top_n = 5

#     # Optional list of stock codes to filter recommendations
#     stock_codes = ["85123A", "71053", "84406B"]

#     # Validate stock codes exist
#     available_stocks = df_clean["StockCode"].unique()
#     valid_stock_codes = [code for code in stock_codes if code in available_stocks]
#     if not valid_stock_codes:
#         print(f"❌ None of the stock codes {stock_codes} found in dataset")
#         print(f"Available stock codes (first 10): {available_stocks[:10]}")
#         return

#     print(f"🔍 Getting recommendations for customer {target_user_id} using Groq API...")
#     print(f"📦 Valid stock codes: {valid_stock_codes}")

#     # Initialize with Groq API (uses API key from crew_agents.py or environment)
#     agents = RecommendationAgents()

#     # Set the dataframe for the tools
#     agents.set_dataframe(df_clean)

#     # Run recommendations with Groq API
#     results = agents.run_recommendations(
#         target_user_id=target_user_id, stock_codes=valid_stock_codes, top_n=top_n
#     )

#     print("\n=== Recommendation Results from Groq API ===")
#     print(results)


if __name__ == "__main__":
    main()
