"""
Configuration settings for the recommendation system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    """API configuration settings."""

    groq_api_key: Optional[str] = None
    google_api_key: str = "AIzaSyAlBLzP3vr560ZqiLyq6wU7pTBcYU74AyY"

    def __post_init__(self):
        # Try to get API keys from environment variables
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")


@dataclass
class ModelConfig:
    """Model configuration settings."""

    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.60
    tfidf_stop_words: str = "english"
    collaborative_filtering_top_n: int = 10
    content_based_top_n: int = 5


@dataclass
class CrewAIConfig:
    """CrewAI configuration settings."""

    llm_model: str = "gemini/gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1000
    verbose: bool = True
    max_rpm: int = 15
    max_iterations: int = 3


@dataclass
class Config:
    """Main configuration class."""

    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    crew: CrewAIConfig = field(default_factory=CrewAIConfig)

    # Dataset settings
    dataset_name: str = "carrie1/ecommerce-data"
    encoding: str = "ISO-8859-1"

    # Output paths
    output_dir: str = "output"
    prepared_data_file: str = "prepared_data.csv"
    recommendations_file: str = "recommendations.json"
