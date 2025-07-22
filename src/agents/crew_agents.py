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
    
    def __init__(self, api_key="AIzaSyCLYT2SLfK8VbHBjXMtr04PhTvN92lJXa8"):
        # Try the format that worked in your tests
        self.llm = LLM(
            model="gemini/gemini-1.5-flash",  # Format that worked
            api_key=api_key.strip(),
            temperature=0.7,
            verbose=False,
            max_tokens=200,
        )
        print("🤖 Using Gemini AI model (gemini/ format)")
        self.llm_available = True
        
        if self.llm_available and self.llm:
            self.data_engineer = Agent(
                role='Data Analysis Expert',
                goal='Analyze retail transaction patterns and provide insights',
                backstory="""You are an expert data analyst who specializes in retail analytics. 
                You analyze customer behavior, transaction patterns, and provide actionable business insights.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
            
            self.cf_specialist = Agent(
                role='Recommendation Analysis Expert', 
                goal='Analyze product recommendations and explain customer preferences',
                backstory="""You are a recommendation system expert who analyzes collaborative filtering results.
                You explain why certain products are recommended and identify customer behavior patterns.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
        else:
            self.data_engineer = None
            self.cf_specialist = None
            print("⚠️ AI agents disabled - running in offline mode")
    
    def run_recommendations(self, df, target_user_id):
        """Run the recommendation process with CrewAI analysis."""
        
        print("🔄 Starting CrewAI analysis with Gemini AI...")
        
        # First, run the actual recommendation logic
        from src.models.recommendations import CollaborativeFiltering
        cf_model = CollaborativeFiltering()
        recommendations = cf_model.get_recommendations(df, target_user_id, top_n=10)
        
        # Get basic statistics
        total_customers = df['CustomerID'].nunique()
        total_products = df['StockCode'].nunique()
        user_transactions = len(df[df['CustomerID'] == target_user_id])
        
        # Get user's country/district
        user_data = df[df['CustomerID'] == target_user_id]
        user_country = user_data['Country'].mode().values[0] if not user_data.empty else 'Unknown'
        user_district = user_data.get('District', pd.Series(['Unknown'])).mode().values[0] if not user_data.empty else 'Unknown'
        
        # Create summary text for agents to analyze
        data_summary = f"""
Dataset Overview:
- Total customers: {total_customers:,}
- Total products: {total_products:,} 
- Total transactions: {len(df):,}
- Target customer {target_user_id}: {user_transactions} transactions from {user_country} ({user_district})
        """
        
        # Format recommendations for analysis
        if recommendations is not None and not recommendations.empty:
            rec_summary = f"""
Collaborative Filtering Recommendations for Customer {target_user_id}:
{recommendations[['StockCode', 'Description', 'Users']].head(8).to_string(index=False)}
            """
        else:
            rec_summary = f"No recommendations could be generated for customer {target_user_id}"
        
        # Create tasks for CrewAI agents
        data_task = Task(
            description=f"""Analyze this e-commerce dataset summary and provide insights:
            
            {data_summary}
            
            Focus on:
            1. Dataset scale and customer engagement levels
            2. Geographic distribution patterns 
            3. Customer transaction behavior
            4. Business implications""",
            agent=self.data_engineer,
            expected_output="Comprehensive analysis of dataset characteristics and customer behavior patterns",
            human_input=False
        )
        
        rec_task = Task(
            description=f"""Analyze these product recommendations and explain the patterns:
            
            {rec_summary}
            
            Focus on:
            1. Product category patterns in recommendations
            2. Customer preference insights
            3. Recommendation quality assessment
            4. Business value of these suggestions""",
            agent=self.cf_specialist,
            expected_output="Detailed analysis of recommendation patterns and customer preferences",
            context=[data_task],
            human_input=False
        )
        
        # Create and run crew
        crew = Crew(
            agents=[self.data_engineer, self.cf_specialist],
            tasks=[data_task, rec_task],
            verbose=True,
            respect_context_window=True
        )
        
        results = crew.kickoff()
        print("✅ CrewAI analysis completed successfully!")
        return results
    
   