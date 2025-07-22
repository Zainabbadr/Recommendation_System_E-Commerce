"""
CrewAI agents for the recommendation system.
"""

import os
import logging
import time
from crewai import Agent, Task, Crew, LLM
from src.data.processor import prepare_data
from src.models.recommendations import collaborative_filtering_recommendations

# Disable OpenTelemetry logging
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


class RecommendationAgents:
    """Class to manage recommendation system agents."""
    
    def __init__(self, api_key="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"):
        self.llm = LLM(
            model="groq/llama-3.1-8b-instant",  # Groq's fast Llama model
            temperature=0.1,  # Very low temperature for minimal tokens
            max_tokens=200,   # Drastically reduced token limit
            verbose=False,
            api_key=api_key,
            timeout=20,       # Shorter timeout
            max_retries=1,    # Only 1 retry to save quota
            request_timeout=10
        )
        
        self.data_engineer = Agent(
            role='Data Analyst',
            goal='Analyze retail data patterns',
            backstory="Expert in e-commerce analytics and customer segmentation",
            verbose=False,
            llm=self.llm,
            allow_delegation=False,
            max_iterations=1,  # Only 1 iteration to save quota
            tools=[prepare_data]
        )
        
        self.cf_specialist = Agent(
            role='Recommender',
            goal='Make recommendations',
            backstory='Recommendation expert',
            verbose=False,
            llm=self.llm,
            allow_delegation=False,
            max_iterations=1,  # Only 1 iteration to save quota
            tools=[collaborative_filtering_recommendations]
        )
    
    def create_tasks(self):
        """Create minimal tasks for the agents."""
        data_prep_task = Task(
            description="Clean data",
            agent=self.data_engineer,
            expected_output="Clean data",
            human_input=False
        )
        
        cf_recommendation_task = Task(
            description="Generate recommendations",
            agent=self.cf_specialist,
            expected_output="Product recommendations",
            context=[data_prep_task],
            human_input=False
        )
        
        return [data_prep_task, cf_recommendation_task]
    
    def create_crew(self):
        """Create and return the crew."""
        tasks = self.create_tasks()
        
        return Crew(
            agents=[self.data_engineer, self.cf_specialist],
            tasks=tasks,
            verbose=False,
            max_rpm=1,      # EXTREMELY conservative - only 1 request per minute
            respect_context_window=True
        )
    
    def run_recommendations(self, df, target_user_id):
        """Run the recommendation process with ultra-conservative quota usage."""
        print("🔄 Starting minimal CrewAI analysis to preserve quota...")
        
        try:
            crew = self.create_crew()
            
            inputs = {
                'df': df,
                'target_user_id': target_user_id
            }
            
            # Single attempt only to preserve quota
            results = crew.kickoff(inputs=inputs)
            print("✅ CrewAI analysis completed successfully!")
            return results
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["429", "quota", "rate limit", "resource_exhausted"]):
                print("⚠️  GROQ QUOTA EXCEEDED")
                print("💡 Groq typically has higher limits - check your API key")
                return self._quota_exceeded_fallback(df, target_user_id)
            elif any(keyword in error_msg for keyword in ["503", "overloaded", "unavailable"]):
                print("⚠️  Groq service temporarily unavailable")
                print("💡 Groq is usually faster - trying fallback analysis")
                return self._quota_exceeded_fallback(df, target_user_id)
            else:
                print(f"❌ API Error: {e}")
                return self._quota_exceeded_fallback(df, target_user_id)
    
    def _quota_exceeded_fallback(self, df, target_user_id):
        """Provide analysis without using API quota."""
        print("🔄 Providing offline analysis to preserve quota...")
        
        try:
            # Get user info without API calls
            user_data = df[df['CustomerID'] == target_user_id]
            if user_data.empty:
                district = "Unknown"
                purchase_count = 0
            else:
                district = user_data['District'].mode().values[0] if len(user_data['District'].mode().values) > 0 else "Unknown"
                purchase_count = len(user_data)
            
            # Simple district analysis
            district_stats = df['District'].value_counts()
            total_customers = df['CustomerID'].nunique()
            total_products = df['StockCode'].nunique()
            
            analysis = f"""
=== OFFLINE ANALYSIS (Quota-Preserving) ===

Target User Analysis:
• User ID: {target_user_id}
• District: {district}
• Purchase History: {purchase_count} transactions
• Status: {'Active' if purchase_count > 0 else 'New/Inactive'} customer

Dataset Overview:
• Total Customers: {total_customers:,}
• Total Products: {total_products:,}
• Transactions: {len(df):,}

District Distribution:
{district_stats.head().to_string()}

Recommendations:
1. Focus on {district} region for similar customers
2. Leverage collaborative filtering within same district
3. Consider top products from {district} market
4. Geographic segmentation is key for this dataset

Note: This analysis preserves your API quota. 
For AI-powered insights, use when quota is available.
            """
            
            return analysis.strip()
            
        except Exception as e:
            return f"Analysis unavailable due to error: {e}"
