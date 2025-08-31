# E-Commerce Recommendation System

A comprehensive, AI-powered recommendation system for e-commerce platforms that combines multiple recommendation approaches including collaborative filtering, content-based filtering, product categorization, and intelligent AI agents using LangGraph and LangChain.

## 🚀 Features

### Core Recommendation Algorithms
- **Collaborative Filtering**: User-based recommendations using cosine similarity and matrix factorization
- **Content-Based Filtering**: Product recommendations based on item descriptions using TF-IDF and semantic embeddings
- **Product Categorization**: Automatic product grouping using sentence transformers and clustering
- **Geographic Segmentation**: District-based user segmentation for location-aware recommendations

### AI-Powered Components
- **LangGraph Chatbot**: Intelligent conversational interface for product recommendations
- **AI Agents**: Automated recommendation workflows using LangChain agents
- **Semantic Search**: Advanced product search using embeddings and similarity matching
- **Personalized Conversations**: Context-aware chat interface for user interactions

### Web Interface
- **Django Web Application**: Full-featured web interface for recommendations
- **RESTful API**: JSON API endpoints for integration with other systems
- **Real-time Processing**: Live recommendation generation and updates
- **User Management**: Customer profile management and history tracking

### Data Processing
- **Automated Data Cleaning**: Intelligent preprocessing for e-commerce transaction data
- **Feature Engineering**: Advanced feature extraction and transformation
- **Data Validation**: Comprehensive data quality checks and validation
- **Scalable Architecture**: Designed to handle large-scale e-commerce datasets

## 📁 Project Structure

```
Recommendation_System_E-Commerce/
├── src/                          # Core source code
│   ├── agents/
│   │   └── crew_agents.py        # AI agents for recommendation workflows
│   ├── chatbot/
│   │   └── langgraph_chatbot.py  # LangGraph-based conversational interface
│   ├── data/
│   │   └── processor.py          # Data loading and preprocessing
│   ├── models/
│   │   └── recommendations.py    # Recommendation algorithms
│   └── utils/
│       └── config.py             # Configuration settings
├── recommendations/              # Django app for web interface
│   ├── templates/               # HTML templates
│   ├── views.py                 # Django views and API endpoints
│   ├── models.py                # Database models
│   └── urls.py                  # URL routing
├── recommendation_frontend/     # Django project settings
├── static/                      # Static files (CSS, JS, images)
├── media/                       # User-uploaded files
├── db/                          # Database files
├── main.py                      # Command-line interface
├── manage.py                    # Django management commands
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker deployment
├── Dockerfile                   # Docker configuration
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Zainabbadr/Recommendation_System_E-Commerce.git
cd Recommendation_System_E-Commerce
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the root directory:
```bash
# API Keys (optional - system works without them)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Django settings
DEBUG=True
SECRET_KEY=your_django_secret_key_here
DATABASE_URL=sqlite:///db/db.sqlite3
```

5. **Initialize the database:**
```bash
python manage.py migrate
python manage.py collectstatic
```

### Docker Installation

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

2. **Access the application:**
- Web interface: http://localhost:8000
- API endpoints: http://localhost:8000/api/

## 🚀 Usage

### Web Interface

1. **Start the Django server:**
```bash
python manage.py runserver
```

2. **Access the web interface:**
- Open http://localhost:8000 in your browser
- Navigate to the recommendations section
- Enter a customer ID to get personalized recommendations

### Command Line Interface

Run the main application for command-line recommendations:
```bash
python main.py
```

### Programmatic Usage

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
recommendations = cf_model.get_recommendations(
    target_user_id=17850, 
    df=clean_df,
    top_n=10
)

# Get content-based recommendations
cb_model.fit(clean_df)
content_recs = cb_model.get_recommendations(
    product_description="white hanging heart t-light holder",
    top_n=5
)

# Use AI agents for advanced recommendations
agents = RecommendationAgents()
agents.set_dataframe(clean_df)
ai_recommendations = agents.run_recommendations(
    target_user_id=17850,
    stock_codes=['product1', 'product2'],
    top_n=5
)
```

### Chatbot Interface

```python
from src.chatbot.langgraph_chatbot import LangGraphChatbot

# Initialize chatbot
chatbot = LangGraphChatbot()

# Start conversation
response = chatbot.chat("I'm looking for home decor items")
print(response)

# Get product recommendations
recommendations = chatbot.get_recommendations("kitchen accessories")
```

## 🔧 Configuration

### Model Configuration

```python
from src.utils.config import Config, ModelConfig

config = Config()

# Adjust recommendation parameters
config.model.similarity_threshold = 0.70
config.model.collaborative_filtering_top_n = 20
config.model.content_based_top_n = 15
config.model.min_interactions = 3

# Configure AI models
config.ai.enable_agents = True
config.ai.enable_chatbot = True
config.ai.max_tokens = 1000
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DEBUG` | Django debug mode | No | `True` |
| `SECRET_KEY` | Django secret key | No | Auto-generated |
| `DATABASE_URL` | Database connection string | No | SQLite |
| `GOOGLE_API_KEY` | Google AI API key | No | None |
| `GEMINI_API_KEY` | Gemini API key | No | None |

## 📊 Dataset

The system uses the "carrie1/ecommerce-data" dataset from Kaggle, which includes:
- Customer transaction data
- Product information
- Geographic data (districts)
- Purchase history and patterns

The dataset is automatically downloaded using `kagglehub` during first run.

### Data Schema

```python
{
    'CustomerID': 'Unique customer identifier',
    'StockCode': 'Product stock code',
    'Description': 'Product description',
    'Quantity': 'Purchase quantity',
    'UnitPrice': 'Product unit price',
    'Country': 'Customer country',
    'District': 'Customer district/region',
    'InvoiceDate': 'Purchase timestamp'
}
```

## 🔌 API Reference

### REST API Endpoints

#### Get Recommendations
```http
GET /api/recommendations/{customer_id}/
```

**Parameters:**
- `customer_id` (int): Target customer ID
- `top_n` (int, optional): Number of recommendations (default: 10)
- `method` (str, optional): 'collaborative', 'content', or 'hybrid' (default: 'hybrid')

**Response:**
```json
{
    "customer_id": 17850,
    "recommendations": [
        {
            "stock_code": "ABC123",
            "description": "Product Description",
            "score": 0.85,
            "reason": "Similar to your previous purchases"
        }
    ],
    "method": "hybrid",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Search Products
```http
GET /api/search/
```

**Parameters:**
- `query` (str): Search query
- `top_n` (int, optional): Number of results (default: 10)

#### Chat Interface
```http
POST /api/chat/
```

**Request Body:**
```json
{
    "message": "I'm looking for kitchen accessories",
    "customer_id": 17850
}
```

**Response:**
```json
{
    "response": "I found some great kitchen accessories for you!",
    "recommendations": [...],
    "conversation_id": "conv_123"
}
```

## 🐳 Deployment

### Docker Deployment

1. **Build the image:**
```bash
docker build -t recommendation-system .
```

2. **Run with Docker Compose:**
```bash
docker-compose up -d
```

3. **Scale the application:**
```bash
docker-compose up -d --scale web=3
```

### Production Deployment

1. **Set production environment variables:**
```bash
export DEBUG=False
export SECRET_KEY=your_production_secret_key
export DATABASE_URL=postgresql://user:pass@host:port/db
```

2. **Run migrations:**
```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

3. **Start with Gunicorn:**
```bash
gunicorn recommendation_frontend.wsgi:application --bind 0.0.0.0:8000
```

## 🧪 Testing

### Run Tests
```bash
python manage.py test
```

### Test Individual Components
```bash
# Test recommendation algorithms
python -m pytest tests/test_recommendations.py

# Test data processing
python -m pytest tests/test_processor.py

# Test API endpoints
python -m pytest tests/test_api.py
```

## 📈 Performance

### Optimization Tips

1. **Database Indexing:**
```sql
CREATE INDEX idx_customer_stock ON transactions(CustomerID, StockCode);
CREATE INDEX idx_stock_description ON products(StockCode, Description);
```

2. **Caching:**
```python
# Enable Redis caching
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

3. **Model Optimization:**
- Use batch processing for large datasets
- Implement lazy loading for embeddings
- Cache similarity matrices

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch:**
```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes and commit:**
```bash
git commit -m 'Add amazing feature'
```

4. **Push to the branch:**
```bash
git push origin feature/amazing-feature
```

5. **Create a Pull Request**

### Development Setup

1. **Install development dependencies:**
```bash
pip install -r requirements-dev.txt
```

2. **Set up pre-commit hooks:**
```bash
pre-commit install
```

3. **Run linting:**
```bash
flake8 src/
black src/
isort src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for providing the e-commerce dataset
- LangChain and LangGraph communities for AI framework
- Django community for the web framework
- All contributors and users of this system

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/Zainabbadr/Recommendation_System_E-Commerce/issues)
- **Documentation:** [Wiki](https://github.com/Zainabbadr/Recommendation_System_E-Commerce/wiki)
- **Email:** support@recommendation-system.com

## 🔄 Changelog

### Version 2.0.0 (Latest)
- Added LangGraph chatbot interface
- Improved AI agent capabilities
- Enhanced web interface with Django
- Added comprehensive API endpoints
- Improved performance and scalability

### Version 1.0.0
- Initial release with basic recommendation algorithms
- Collaborative and content-based filtering
- Basic command-line interface