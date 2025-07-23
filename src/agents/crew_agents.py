"""
CrewAI agents for the recommendation system.
"""

import os
import logging
import time
import pandas as pd
from crewai import Agent, Task, Crew, LLM
from src.data.processor import prepare_data
from src.models.recommendations import collaborative_filtering_recommendations

# Disable OpenTelemetry logging
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


class RecommendationAgents:
    """Class to manage recommendation system agents."""
    
    def __init__(self, api_key=None):
        if api_key and api_key.strip():
            # Use Gemini with provided API key
            try:
                self.llm = LLM(
                    model="gemini/gemini-1.5-flash",  # Format that worked
                    api_key=api_key.strip(),
                    temperature=0.7,
                    verbose=False,
                    max_tokens=200,
                )
                print("🤖 Using Gemini AI model (gemini/ format)")
                self.llm_available = True
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                print("💡 Get your API key from: https://aistudio.google.com/app/apikey")
                self.llm = None
                self.llm_available = False
        else:
            print("⚠️ No Gemini API key provided")
            print("💡 Get your API key from: https://aistudio.google.com/app/apikey")
            self.llm = None
            self.llm_available = False
        
        if self.llm_available and self.llm:
            self.data_engineer = Agent(
                role='Retail Data Specialist',
                goal='Prepare transaction data for recommendation analysis',
                backstory="""Expert in cleaning and transforming retail transaction data
                with special focus on geographic segmentation""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False,
                max_iterations=3,  # Prevent infinite loops
                tools=[prepare_data]
            )
            
            self.cf_specialist = Agent(
                role='Collaborative Filtering Specialist',
                goal='Generate personalized product recommendations',
                backstory='Data scientist specializing in neighborhood-based recommendation systems',
                verbose=True,
                llm=self.llm,
                tools=[collaborative_filtering_recommendations]
            )
        else:
            self.data_engineer = None
            self.cf_specialist = None
            print("⚠️ AI agents disabled - running in offline mode")
    
    def create_tasks(self, df_json, target_user_id):
        """Create tasks with embedded data parameters."""
        data_prep_task = Task(
            description="""Prepare retail transaction data by:
            1. Handling missing values appropriately
            2. Ensuring correct data types
            3. Validating district information""",
            agent=self.data_engineer,
            expected_output="""Cleaned DataFrame with:
            - Validated districts (Europe, Oceania, Asia-Pacific, North America, Middle East, Unknown, Africa, South America)
            - Proper data types
            - Missing values handled""",
            output_file="prepared_data.csv",
            human_input=False
        )
        
        cf_recommendation_task = Task(
            description="""Generate recommendations using existing collaborative filtering function:
            1. Process only validated district data
            2. Apply user-based collaborative filtering
            3. Return top 10 recommendations per user
            4. Include similarity metrics""",
            agent=self.cf_specialist,
            expected_output="""Dictionary containing:
            - target_user_id
            - district
            - Description of product
            - top_10_recommendations
            - similar_users_used
            - similarity_scores""",
            context=[data_prep_task],
            output_file="recommendations.json",
            human_input=False
        )
        
        return [data_prep_task, cf_recommendation_task]
    
    def create_crew(self, df_json, target_user_id):
        """Create and return the crew with embedded data."""
        tasks = self.create_tasks(df_json, target_user_id)
        
        return Crew(
            agents=[self.data_engineer, self.cf_specialist],
            tasks=tasks,
            verbose=False,
            respect_context_window=True
        )
    
    def run_recommendations(self, df, target_user_id):
        """Run the recommendation process with JSON-serializable DataFrame."""
        
        # Check if agents are available
        if not self.llm_available or not self.data_engineer or not self.cf_specialist:
            print("⚠️ Gemini AI not available - using offline analysis")
            return self._analysis_fallback(df, target_user_id)
        
        print("🔄 Starting CrewAI analysis with Gemini AI...")
        
        try:
            # Convert sampled DataFrame to JSON
            df_json = df.to_json(orient='records')
            print(f"JSON data length: {len(df_json):,} characters")
            
            # Create crew with embedded data
            crew = self.create_crew(df_json, target_user_id)
            
            # Simple inputs for tools to use
            inputs = {
                'df_json': df_json,
                'target_user_id': int(target_user_id),
                'top_n': 10
            }
                        
            results = crew.kickoff(inputs=inputs)
            print("✅ CrewAI analysis completed successfully!")
            return results
            
        except Exception as e:
            error_msg = str(e).lower()
            if "invalid api key" in error_msg or "authentication" in error_msg:
                print("❌ Gemini API Key Error: Please check your API key")
                print("💡 Get a valid API key from: https://aistudio.google.com/app/apikey")
            elif "groq" in error_msg:
                print("❌ System trying to use Groq instead of Gemini")
                print("💡 This suggests a configuration issue - using offline mode")
            else:
                print(f"❌ Error during analysis: {e}")
            return self._analysis_fallback(df, target_user_id)
    
    def _analysis_fallback(self, df, target_user_id):
        """Provide analysis without using the LLM."""
        print("🔄 Providing offline analysis due to error...")
        
        try:
            # Get user info without API calls
            user_data = df[df['CustomerID'] == target_user_id]
            if user_data.empty:
                country = "Unknown"
                purchase_count = 0
            else:
                country = user_data['Country'].mode().values[0] if len(user_data['Country'].mode().values) > 0 else "Unknown"
                purchase_count = len(user_data)
            
            # Simple country analysis
            country_stats = df['Country'].value_counts()
            total_customers = df['CustomerID'].nunique()
            total_products = df['StockCode'].nunique()
            
            analysis = f"""
=== OFFLINE ANALYSIS (Fallback) ===

Target User Analysis:
• User ID: {target_user_id}
• Country: {country}
• Purchase History: {purchase_count} transactions
• Status: {'Active' if purchase_count > 0 else 'New/Inactive'} customer

Dataset Overview:
• Total Customers: {total_customers:,}
• Total Products: {total_products:,}
• Transactions: {len(df):,}

Country Distribution (Top 5):
{country_stats.head().to_string()}

Recommendations:
1. Focus on {country} market for similar customers
2. Leverage collaborative filtering within same country
3. Consider top products from {country} market
4. Geographic segmentation is key for this dataset

Note: This is a fallback analysis due to missing API key or AI system error.
            """
            
            return analysis.strip()
            
        except Exception as e:
            return f"Analysis unavailable due to error: {e}"
    
   