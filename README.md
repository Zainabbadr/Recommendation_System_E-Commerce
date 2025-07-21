# E-Commerce Recommendation System

A comprehensive recommendation system for e-commerce that combines multiple approaches including collaborative filtering, content-based filtering, product categorization, and AI-powered agents using CrewAI.

## Features

- **Data Processing**: Automated data cleaning and preprocessing for e-commerce transaction data
- **Collaborative Filtering**: User-based recommendations using cosine similarity 
- **Content-Based Filtering**: Product recommendations based on item descriptions using TF-IDF
- **Product Categorization**: Automatic product grouping using sentence transformers
- **AI Agents**: CrewAI-powered agents for automated recommendation workflows
- **Geographic Segmentation**: District-based user segmentation for better recommendations

## Project Structure

```
Recommendation_System_E-Commerce/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── processor.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── recommendations.py    # Recommendation algorithms
│   ├── agents/
│   │   ├── __init__.py
│   │   └── crew_agents.py        # CrewAI agents
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Configuration settings
├── main.py                       # Main application entry point
├── setup.py                      # Setup and installation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zainabbadr/Recommendation_System_E-Commerce.git
cd Recommendation_System_E-Commerce
```

2. Run the setup script:
```bash
python setup.py
```

Or manually install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main application:
```bash
python main.py
```

### Using Individual Components

```python
from src.data.processor import DataProcessor
from src.models.recommendations import CollaborativeFiltering, ContentBasedFiltering
from src.agents.crew_agents import RecommendationAgents

# Initialize components
processor = DataProcessor()
cf_model = CollaborativeFiltering()
cb_model = ContentBasedFiltering()

# Load and process data
df = processor.load_dataset()
clean_df = processor.clean_data(df)

# Get collaborative filtering recommendations
cf_model.fit(clean_df)
recommendations = cf_model.get_recommendations(target_user_id=17850, df=clean_df)

# Get content-based recommendations
cb_model.fit(clean_df)
content_recs = cb_model.get_recommendations("white hanging heart t-light holder")
```

### Configuration

You can customize the system behavior by modifying the configuration:

```python
from src.utils.config import Config, ModelConfig

config = Config()
config.model.similarity_threshold = 0.70  # Adjust similarity threshold
config.model.collaborative_filtering_top_n = 20  # More recommendations
```

## API Keys

The system uses Google Gemini API for CrewAI agents. Set your API key:

1. In the configuration file: `src/utils/config.py`
2. As an environment variable: `export GOOGLE_API_KEY="your-api-key"`

## Dataset

The system uses the "carrie1/ecommerce-data" dataset from Kaggle, which is automatically downloaded using kagglehub.

## Dependencies

Key dependencies include:
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- sentence-transformers: Text embeddings for categorization
- crewai: AI agent framework
- kagglehub: Dataset access

See `requirements.txt` for the complete list.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).