"""
CrewAI agents for the recommendation system.
"""

import os
import logging
from crewai import Agent, Task, Crew, LLM
from src.data.processor import prepare_data
from src.models.recommendations import collaborative_filtering_recommendations

# Disable OpenTelemetry logging
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


class RecommendationAgents:
    """Class to manage recommendation system agents."""
    
    def __init__(self, api_key="AIzaSyAlBLzP3vr560ZqiLyq6wU7pTBcYU74AyY"):
        self.llm = LLM(
            model="gemini/gemini-1.5-flash",
            temperature=0.7,
            max_tokens=1000,
            verbose=True,
            api_key=api_key,
        )
        
        self.data_engineer = Agent(
            role='Retail Data Specialist',
            goal='Prepare transaction data for recommendation analysis',
            backstory="""Expert in cleaning and transforming retail transaction data
            with special focus on geographic segmentation""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iterations=3,
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
    
    def create_tasks(self):
        """Create tasks for the agents."""
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
    
    def create_crew(self):
        """Create and return the crew."""
        tasks = self.create_tasks()
        
        return Crew(
            agents=[self.data_engineer, self.cf_specialist],
            tasks=tasks,
            verbose=True,
            max_rpm=15,
            respect_context_window=True,
            memory=False  # Disable memory to avoid ChromaDB issues
        )
    
    def run_recommendations(self, df, target_user_id):
        """Run the recommendation process."""
        crew = self.create_crew()
        
        inputs = {
            'df': df,
            'target_user_id': target_user_id
        }
        
        results = crew.kickoff(inputs=inputs)
        return results
