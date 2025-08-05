"""
CrewAI agents for the recommendation system using Groq API with Llama 3.3.
"""

import os
import logging
import time
import pandas as pd
import json
from typing import Union, List, Tuple, Dict
from langchain.schema import AgentFinish
from crewai import Agent, Task, Crew, LLM, Process

# from langchain_groq import ChatGroq
# from langchain_community.tools import DuckDuckGoSearchRun
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List

# Fix backports issue
import sys

if "backports" in sys.modules:
    del sys.modules["backports"]

# Disable OpenTelemetry logging
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

# Global variables for callback logging
agent_finishes = []
call_number = 0


def print_agent_output(
    agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish],
    agent_name: str = "Generic call",
):
    global call_number
    call_number += 1
    with open("crew_callback_logs.txt", "a") as log_file:
        # Try to parse the output if it is a JSON string
        if isinstance(agent_output, str):
            try:
                agent_output = json.loads(agent_output)
            except json.JSONDecodeError:
                pass

        # Check if the output is a list of tuples
        if isinstance(agent_output, list) and all(
            isinstance(item, tuple) for item in agent_output
        ):
            print(
                f"-{call_number}----Dict------------------------------------------",
                file=log_file,
            )
            for action, description in agent_output:
                print(f"Agent Name: {agent_name}", file=log_file)
                print(f"Tool used: {getattr(action, 'tool', 'Unknown')}", file=log_file)
                print(
                    f"Tool input: {getattr(action, 'tool_input', 'Unknown')}",
                    file=log_file,
                )
                print(f"Action log: {getattr(action, 'log', 'Unknown')}", file=log_file)
                print(f"Description: {description}", file=log_file)
                print(
                    "--------------------------------------------------", file=log_file
                )

        # Check if the output is AgentFinish
        elif isinstance(agent_output, AgentFinish):
            print(
                f"-{call_number}----AgentFinish---------------------------------------",
                file=log_file,
            )
            print(f"Agent Name: {agent_name}", file=log_file)
            agent_finishes.append(agent_output)
            output = agent_output.return_values
            print(f"AgentFinish Output: {output['output']}", file=log_file)
            print("--------------------------------------------------", file=log_file)

        # Handle unexpected formats
        else:
            print(f"-{call_number}-Unknown format of agent_output:", file=log_file)
            print(type(agent_output), file=log_file)
            print(agent_output, file=log_file)


class RecommendationInput(BaseModel):
    target_user_id: int = Field(..., description="The ID of the target user")
    stock_codes: List[str] = Field(
        ..., description="List of stock codes (products) purchased by user"
    )
    top_n: int = Field(10, description="Number of recommendations to return")


class CollaborativeFilteringTool(BaseTool):
    name: str = "collaborative_filtering_recommendations"
    description: str = (
        "Generate product recommendations using district-aware collaborative filtering"
    )
    args_schema: Type[BaseModel] = RecommendationInput

    def _run(
        self, target_user_id: int, stock_codes: list[str], top_n: int = 10
    ) -> dict:
        try:
            # This will be set by the main application
            if not hasattr(self, "df") or self.df is None:
                return {
                    "user_id": target_user_id,
                    "status": "DataFrame not available",
                    "recommendations": [],
                }

            if target_user_id not in self.df["CustomerID"].unique():
                return {
                    "user_id": target_user_id,
                    "status": f"CustomerID {target_user_id} not found",
                    "recommendations": [],
                }

            invalid_stock_codes = [
                code
                for code in stock_codes
                if code not in self.df["StockCode"].unique()
            ]
            if invalid_stock_codes:
                return {
                    "user_id": target_user_id,
                    "status": f"Stock codes not found: {invalid_stock_codes}",
                    "recommendations": [],
                }

            user_item_matrix = self.df.pivot_table(
                index="CustomerID",
                columns="StockCode",
                values="Quantity",
                aggfunc="mean",
            ).fillna(0)

            # Current and added items
            current_items = set(
                user_item_matrix.loc[target_user_id][
                    user_item_matrix.loc[target_user_id] > 0
                ].index
            )
            updated_items = current_items.union(set(stock_codes))
            for item in updated_items:
                user_item_matrix.loc[target_user_id, item] = 1

            target_district = (
                self.df[self.df["CustomerID"] == target_user_id]["District"]
                .mode()
                .values[0]
            )
            district_users = self.df[
                (self.df["District"] == target_district)
                & (self.df["CustomerID"] != target_user_id)
            ]["CustomerID"].unique()
            district_users = [
                uid for uid in district_users if uid in user_item_matrix.index
            ]

            filtered_matrix = user_item_matrix.loc[[target_user_id] + district_users]
            similarities = cosine_similarity(filtered_matrix)[0][1:]

            similar_users = sorted(
                zip(district_users, similarities), key=lambda x: x[1], reverse=True
            )[:10]

            recommendations = defaultdict(lambda: {"score": 0, "users": []})
            for similar_user, similarity in similar_users:
                similar_items = user_item_matrix.loc[similar_user]
                for item in similar_items[similar_items > 0].index:
                    if item not in updated_items:
                        recommendations[item]["score"] += similarity
                        recommendations[item]["users"].append(similar_user)

            if not recommendations:
                return {
                    "user_id": target_user_id,
                    "district": target_district,
                    "status": "No new recommendations found",
                    "recommendations": [],
                }

            result = []
            for item, data in sorted(
                recommendations.items(), key=lambda x: x[1]["score"], reverse=True
            )[:top_n]:
                item_info = self.df[self.df["StockCode"] == item].iloc[0]
                result.append(
                    {
                        "stock_code": item,
                        "description": item_info["Description"],
                        "price": round(item_info["UnitPrice"], 2),
                        "recommended_by": data["users"],
                    }
                )

            # Get prices of user's items (original and added)
            user_items_info = self.df[self.df["StockCode"].isin(updated_items)]
            user_items_summary = (
                user_items_info.groupby("StockCode")
                .agg({"Description": "first", "UnitPrice": "mean"})
                .reset_index()
            )

            user_items_list = user_items_summary.to_dict(orient="records")
            for item in user_items_list:
                item["UnitPrice"] = round(item["UnitPrice"], 2)

            return {
                "user_id": target_user_id,
                "district": target_district,
                "status": "success",
                "user_items": user_items_list,
                "recommendations": result,
            }

        except Exception as e:
            return {
                "user_id": target_user_id,
                "status": f"Error: {str(e)}",
                "recommendations": [],
            }


class ContentRecommendationInput(BaseModel):
    target_user_id: int = Field(..., description="The ID of the target user")
    stock_codes: List[str] = Field(
        ..., description="List of stock codes for content-based recommendation"
    )
    recommendations_per_stock: int = Field(
        3, description="Number of unique recommendations per stock code"
    )


class ContentBasedRecommendationTool(BaseTool):
    name: str = "content_based_recommendations"
    description: str = "Generate product recommendations using content similarity on product descriptions"
    args_schema: Type[BaseModel] = ContentRecommendationInput

    def _run(
        self,
        target_user_id: int,
        stock_codes: List[str],
        recommendations_per_stock: int = 3,
    ) -> dict:
        try:
            if not hasattr(self, "df") or self.df is None:
                return {
                    "target_user_id": target_user_id,
                    "status": "DataFrame not available",
                    "results": [],
                }

            # Vectorize descriptions once
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(self.df["Description"])

            # Initialize recommendation tracking
            all_seen_descriptions = set()
            all_results = []
            user_items_info = []

            for stock in stock_codes:
                indices = self.df.index[self.df["StockCode"] == stock].tolist()
                if not indices:
                    all_results.append(
                        {
                            "stock_code": stock,
                            "status": f"Stock code '{stock}' not found",
                            "recommendations": [],
                        }
                    )
                    continue

                # Get the first occurrence
                input_idx = indices[0]
                input_vector = tfidf_matrix[input_idx]
                similarity_scores = cosine_similarity(
                    input_vector, tfidf_matrix
                ).flatten()

                input_description = self.df.at[input_idx, "Description"]
                input_unit_price = self.df.at[input_idx, "UnitPrice"]
                user_items_info.append(
                    {
                        "stock_code": stock,
                        "description": input_description,
                        "unit_price": input_unit_price,
                    }
                )

                seen_descriptions = set(all_seen_descriptions)
                seen_descriptions.add(input_description)

                sorted_indices = similarity_scores.argsort()[::-1]
                recommendations = []

                for idx in sorted_indices:
                    if idx == input_idx:
                        continue
                    desc = self.df.at[idx, "Description"]
                    if desc not in seen_descriptions:
                        recommendations.append(
                            {
                                "stock_code": self.df.at[idx, "StockCode"],
                                "description": desc,
                                "unit_price": self.df.at[idx, "UnitPrice"],
                            }
                        )
                        seen_descriptions.add(desc)
                        all_seen_descriptions.add(desc)

                    if len(recommendations) >= recommendations_per_stock:
                        break

                status = (
                    "success" if recommendations else "No similar recommendations found"
                )

                all_results.append(
                    {
                        "stock_code": stock,
                        "status": status,
                        "recommendations": recommendations,
                    }
                )

            return {
                "target_user_id": target_user_id,
                "target_user_items": user_items_info,
                "results": all_results,
            }

        except Exception as e:
            return {
                "target_user_id": target_user_id,
                "status": f"Error: {str(e)}",
                "results": [],
            }


class RecommendationAgents:
    """Class to manage recommendation system agents using Groq API with Llama 3.3."""

    def __init__(
        self,
        groq_api_key=None,
        gemini_api_key=None,
    ):
        """
        Initialize with Groq API and optional Gemini API.

        Args:
            groq_api_key: API key for Groq (optional, can be set via environment)
            gemini_api_key: API key for Gemini (optional, can be set via environment)
        """
        try:
            # Set up Groq LLM
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key

            self.groq_llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                max_completion_tokens=1024,
                top_p=0.9,
                stop=None,
                stream=False,
            )

            # Set up Gemini LLM (optional)
            if gemini_api_key:
                os.environ["Recomm_Gemini"] = gemini_api_key

            self.gemini_llm = LLM(
                model="gemini/gemini-2.5-flash",
                max_completion_tokens=1024,
                top_p=0.9,
                stop=None,
                stream=False,
            )

            print("🤖 Using Groq API with Llama 3.3 model")
            self.llm_available = True

            # Initialize tools
            self.cf_tool = CollaborativeFilteringTool()
            self.cb_tool = ContentBasedRecommendationTool()

            # Create agents
            self.collaborative_recommendation_agent = Agent(
                role="Collaborative Filtering Specialist",
                goal="Generate personalized product recommendations",
                backstory="Data scientist specializing in neighborhood-based recommendation systems",
                llm=self.groq_llm,
                verbose=True,
                allow_delegation=False,
                max_iter=5,
                memory=False,  # Disabled memory to avoid onnxruntime dependency
                step_callback=lambda x: print_agent_output(x, "Recommendation Agent"),
                tools=[self.cf_tool],
            )

            self.content_based_agent = Agent(
                role="Content-Based Recommendation Specialist",
                goal="Generate product recommendations based on content similarity of product descriptions.",
                backstory="You are an AI expert leveraging product description similarity to recommend related items.",
                llm=self.groq_llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3,
                memory=False,  # Disabled memory to avoid onnxruntime dependency
                tools=[self.cb_tool],
            )

            self.reranker_agent = Agent(
                role="Recommendation Reranker",
                goal="Rerank product recommendations based on user fit using both collaborative and content-based results",
                backstory=(
                    "You are an intelligent recommendation assistant that combines different recommendation strategies "
                    "to ensure the most relevant products are returned to users."
                ),
                llm=self.groq_llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3,
                memory=False,  # Disabled memory to avoid onnxruntime dependency
            )

        except Exception as e:
            print(f"⚠️ Error initializing Groq LLM: {e}")
            self.groq_llm = None
            self.gemini_llm = None
            self.llm_available = False
            self.collaborative_recommendation_agent = None
            self.content_based_agent = None
            self.reranker_agent = None
            print("⚠️ AI agents disabled - Groq LLM not available")

    def create_tasks(self, target_user_id, stock_codes, top_n=10):
        """Create tasks for the multi-agent recommendation system."""

        collaborative_recommendation_task = Task(
            description=f"""Generate personalized recommendations for customer {target_user_id} and take input {stock_codes}:
            1. Filter to customers in the same district.
            2. Price of each item.
            3. Return top {top_n} recommendations with scores.""",
            expected_output="""A structured report containing:
            - User ID and district.
            - List of recommended products with:
              * Stock code and description.
              * Unit price.
              * IDs of recommending users.
            - Add your suggestion for the best recommended product.""",
            agent=self.collaborative_recommendation_agent,
        )

        content_based_task = Task(
            description=f"""Generate content-based recommendations for a target user {target_user_id}.
            For each input stock code in the list {stock_codes}, return the top 3 most similar products based on description similarity.
            Ensure there are no duplicate descriptions in the final recommendation list.""",
            expected_output="""A structured report containing:
            - Target user ID.
            - Input stock codes.
            - For each input stock code:
              * List of up to 3 recommended products with:
                - Stock code
                - Description
                - Unit price
            - All descriptions must be unique across recommendations.""",
            agent=self.content_based_agent,
        )

        rerank_task = Task(
            description="""
You are a recommendation reranker. You receive **two JSON inputs**:

1. Collaborative Filtering results, structured as:
{
  "User ID": int,
  "District": string,
  "Recommended Products": [
    {
      "Stock Code": string,
      "Description": string,
      "Unit Price": float,
      "Recommended By": [int]
    },
    ...
  ],
  "Best Recommended Product": { ... }
}

2. Content-Based Filtering results, structured as:
{
  "target_user_id": int,
  "target_user_items": [
    {
      "stock_code": string,
      "description": string,
      "unit_price": float
    }
  ],
  "results": [
    {
      "stock_code": string,
      "status": "success",
      "recommendations": [
        {
          "stock_code": string,
          "description": string,
          "unit_price": float
        }
      ]
    }
  ]
}

Your goal is to:

- Combine and analyze both recommendation lists
- REMOVE ALL DUPLICATES based on Stock Code (each Stock Code should appear only ONCE)
- **CRITICAL: ONLY include recommendations that have ALL required information:**
  * Valid Stock Code (not empty)
  * Valid Description (not empty, not "Not Available", not "N/A")
  * Valid Unit Price (greater than 0.0)
- If any field is missing, empty, or invalid, EXCLUDE that recommendation entirely
- Prioritize items appearing in both lists (mark as "Both")
- Return EXACTLY 5 UNIQUE and COMPLETE recommended products

CRITICAL RULES:
1. NO DUPLICATE Stock Codes allowed
2. NO recommendations with missing/invalid data
3. Description must be a real product description (not "Not Available")
4. Unit Price must be > 0.0
5. If fewer than 5 valid recommendations exist, return only the valid ones
6. Each product should appear only once in the final list

{
  "Top Recommendations": [
    {
      "Stock Code": string,
      "Description": string,
      "Unit Price": float,
      "Source": "Collaborative Filtering" | "Content-Based" | "Both",
      "Popularity": int  // number of recommenders or 0 if from CB only
    },
    ...
  ]
}

Format your output as valid JSON ONLY.
""",
            expected_output="""
A JSON object with "Top Recommendations" containing ONLY complete, valid recommendations. Each product must have:
- Stock Code (string, not empty)
- Description (string, not empty, not "Not Available")
- Unit Price (float, greater than 0.0)
- Source (string: "Collaborative Filtering", "Content-Based", or "Both")
- Popularity (integer)

Return only recommendations with complete, valid data. If fewer than 5 valid recommendations exist, return only the valid ones.
""",
            agent=self.reranker_agent,
            context=[collaborative_recommendation_task, content_based_task],
        )

        return [collaborative_recommendation_task, content_based_task, rerank_task]

    def create_crew(self, target_user_id, stock_codes, top_n=10):
        """Create and return the crew with multi-agent parameters."""
        tasks = self.create_tasks(target_user_id, stock_codes, top_n)
        return Crew(
            agents=[
                self.collaborative_recommendation_agent,
                self.content_based_agent,
                self.reranker_agent,
            ],
            tasks=tasks,
            verbose=True,
            process=Process.sequential,
            full_output=True,
        )

    def set_dataframe(self, df):
        """Set the dataframe for the tools to use."""
        # Store dataframe in the agent class and pass to tools when needed
        self.df = df
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self.cf_tool, "df", df)
        object.__setattr__(self.cb_tool, "df", df)

    def run_recommendations(self, target_user_id, stock_codes, top_n=10):
        """Run the recommendation process with Groq API and Llama 3.3.

        Args:
            target_user_id: The customer ID to get recommendations for
            stock_codes: List of stock codes (products) purchased by user
            top_n: Number of recommendations to return
        """

        # Check if agents are available
        if not self.llm_available or not self.collaborative_recommendation_agent:
            print("⚠️ Groq LLM not available")
            return f"Analysis failed for customer {target_user_id}. Error: Groq LLM not available"

        print(
            f"🔄 Starting CrewAI analysis for customer {target_user_id} with Groq API..."
        )

        try:
            # Create crew with multi-agent parameters
            crew = self.create_crew(target_user_id, stock_codes, top_n)

            # Inputs for tools to use
            inputs = {
                "target_user_id": int(target_user_id),
                "stock_codes": stock_codes,
                "top_n": int(top_n),
            }

            results = crew.kickoff(inputs=inputs)
            print("✅ CrewAI analysis with Groq API completed successfully!")
            return results

        except Exception as e:
            print(f"❌ Error during analysis with Groq API: {e}")
            return f"Analysis failed for customer {target_user_id}. Error: {e}"
