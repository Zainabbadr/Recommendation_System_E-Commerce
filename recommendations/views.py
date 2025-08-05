import json
import sys
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.agents.crew_agents import RecommendationAgents
    CREWAI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CrewAI not available: {e}")
    CREWAI_AVAILABLE = False

from src.data.processor import DataProcessor
from src.models.recommendations import weighted_hybrid_recommendations, CollaborativeFiltering
from recommendations.models import Dim_Products  # Import Django model

# Global variables to cache data
_processor = None
_df_clean = None
_agents = None

def get_processor_and_data():
    """Initialize processor and load data if not already done."""
    global _processor, _df_clean, _agents
    
    if _processor is None:
        print("🔧 Initializing data processor...")
        _processor = DataProcessor()
        df = _processor.load_data_from_sqlite()
        if df is not None:
            _df_clean = _processor.clean_data(df)
            print(f"✅ Data loaded: {len(_df_clean)} rows")
        else:
            print("❌ Failed to load dataset")
            return None, None, None
    
    if _agents is None and CREWAI_AVAILABLE:
        print("🤖 Initializing AI agents...")
        try:
            _agents = RecommendationAgents()
            if _df_clean is not None:
                _agents.set_dataframe(_df_clean)
        except Exception as e:
            print(f"⚠️ Failed to initialize AI agents: {e}")
            _agents = None
    
    return _processor, _df_clean, _agents

def get_products_for_dropdown():
    """Get all products from database for dropdown selection with prices."""
    try:
        from django.db import connection
        
        # Use raw SQL to get products with average prices
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    dp.StockCode,
                    dp.Description,
                    AVG(ft.UnitPrice) as avg_price
                FROM recommendations_dim_products dp
                LEFT JOIN recommendations_fact_transactions ft ON dp.StockCode = ft.StockCode_id
                WHERE dp.Description IS NOT NULL 
                AND dp.Description != ''
                GROUP BY dp.StockCode, dp.Description
                HAVING AVG(ft.UnitPrice) IS NOT NULL
                ORDER BY dp.Description
            """)
            
            rows = cursor.fetchall()
            
            # Format for dropdown: [(stock_code, description_with_price, price), ...]
            product_choices = []
            for row in rows:
                stock_code, description, avg_price = row
                price = float(avg_price) if avg_price else 0.0
                formatted_description = f"{description} ({stock_code}) - ${price:.2f}"
                product_choices.append((stock_code, formatted_description, price))
        
        print(f"📦 Loaded {len(product_choices)} products for dropdown with prices")
        return product_choices
        
    except Exception as e:
        print(f"❌ Error loading products with prices: {e}")
        # Fallback to products without prices
        try:
            products = Dim_Products.objects.exclude(
                Description__isnull=True
            ).exclude(
                Description__exact=''
            ).order_by('Description')
            
            # Format without prices
            product_choices = [
                (product.StockCode, f"{product.Description} ({product.StockCode})", 0.0)
                for product in products
            ]
            
            print(f"📦 Loaded {len(product_choices)} products for dropdown (without prices)")
            return product_choices
            
        except Exception as fallback_error:
            print(f"❌ Fallback error: {fallback_error}")
            return []

def get_simple_recommendations(df, target_user_id, stock_codes, top_n=7):
    """Simple recommendation algorithm as fallback."""
    try:
        print(f"🔄 Using simple recommendation algorithm...")
        
        # Get user's purchase history
        user_purchases = df[df['CustomerID'] == target_user_id]
        if len(user_purchases) == 0:
            print(f"⚠️ No purchase history found for customer {target_user_id}")
            return []
        
        # Get products from similar customers
        user_country = user_purchases['Country'].iloc[0] if 'Country' in df.columns else 'Unknown'
        
        # Find customers from same country
        similar_customers = df[df['Country'] == user_country]['CustomerID'].unique()
        print(f"📊 Found {len(similar_customers)} customers from {user_country}")
        
        # Get popular products among similar customers
        similar_purchases = df[df['CustomerID'].isin(similar_customers)]
        
        # Calculate product popularity and average price
        product_stats = similar_purchases.groupby('StockCode').agg({
            'Quantity': 'sum',
            'UnitPrice': 'mean',
            'Description': 'first'
        }).reset_index()
        
        # Sort by quantity (popularity)
        product_stats = product_stats.sort_values('Quantity', ascending=False)
        
        # Filter out products the user already bought
        user_bought_stocks = set(user_purchases['StockCode'].unique())
        product_stats = product_stats[~product_stats['StockCode'].isin(user_bought_stocks)]
        
        # Get top recommendations
        recommendations = []
        for _, row in product_stats.head(top_n).iterrows():
            recommendations.append({
                'stock_code': row['StockCode'],
                'description': row['Description'] or 'No description',
                'unit_price': round(float(row['UnitPrice']), 2),
                # 'score': float(row['Quantity'])
            })
        
        print(f"✅ Generated {len(recommendations)} simple recommendations")
        return recommendations
        
    except Exception as e:
        print(f"❌ Simple recommendation error: {e}")
        return []
    
# Add this import at the top
from src.chatbot.langgraph_chatbot import RecommendationChatbot
import os

# Global chatbot instance
_chatbot = None

def get_chatbot():
    """Initialize chatbot if not already done."""
    global _chatbot
    
    if _chatbot is None:
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            print("🤖 Initializing chatbot...")
            _chatbot = RecommendationChatbot()
            print("✅ Chatbot initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            _chatbot = None
    
    return _chatbot

def chatbot_view(request):
    """Handle chatbot API requests."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            user_id = data.get('user_id', 'default_user')
            
            if not message.strip():
                return JsonResponse({
                    'error': 'Message cannot be empty',
                    'status': 'error'
                }, status=400)
            
            # Get chatbot instance
            chatbot = get_chatbot()
            
            if chatbot is None:
                return JsonResponse({
                    'response': 'Sorry, the AI chatbot is currently unavailable. Please try the manual recommendation tabs.',
                    'status': 'fallback'
                })
            
            # Get chatbot response
            response = chatbot.chat(message, user_id)
            
            return JsonResponse({
                'response': response,
                'status': 'success'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON data',
                'status': 'error'
            }, status=400)
        except Exception as e:
            print(f"❌ Chatbot error: {e}")
            return JsonResponse({
                'response': 'I encountered an unexpected error. Please try again or use the manual recommendation options.',
                'status': 'error'
            })
    
    return JsonResponse({
        'error': 'Only POST requests allowed',
        'status': 'error'
    }, status=405)

def index(request):
    """Main page with dual-tab recommendation form."""
    # Get products for dropdown (always needed for GET and POST)
    available_products = get_products_for_dropdown()
    
    if request.method == 'POST':
        # Get form data
        target_user_id = request.POST.get('target_user_id')
        selected_stock_codes = request.POST.getlist('selected_products')  # Changed to getlist for multiple selection
        recommendation_type = request.POST.get('recommendation_type', 'ai')
        top_n = 7  # Always 7 recommendations
        
        try:
            target_user_id = int(target_user_id)
            
            # Validate that products were selected
            if not selected_stock_codes:
                return render(request, 'recommendations/index.html', {
                    'error': 'Please select at least one product',
                    'recommendation_type': recommendation_type,
                    'available_products': available_products
                })
            
            # Get data
            processor, df_clean, agents = get_processor_and_data()
            if df_clean is None:
                return render(request, 'recommendations/index.html', {
                    'error': 'Failed to load data',
                    'recommendation_type': recommendation_type,
                    'available_products': available_products
                })
            
            print(f"📊 DataFrame columns: {list(df_clean.columns)}")
            print(f"📊 DataFrame shape: {df_clean.shape}")
            print(f"📊 Sample CustomerIDs: {df_clean['CustomerID'].unique()[:10]}")
            print(f"📦 Selected products: {selected_stock_codes}")
            
            # Validate customer exists
            if target_user_id not in df_clean['CustomerID'].unique():
                available_customers = df_clean['CustomerID'].unique()[:10].tolist()
                return render(request, 'recommendations/index.html', {
                    'error': f'Customer {target_user_id} not found',
                    'available_customers': available_customers,
                    'recommendation_type': recommendation_type,
                    'available_products': available_products
                })
            
            # Validate selected stock codes exist in data
            available_stocks = df_clean['StockCode'].unique()
            valid_stock_codes = [code for code in selected_stock_codes if code in available_stocks]
            
            if not valid_stock_codes:
                return render(request, 'recommendations/index.html', {
                    'error': f'Selected products not found in transaction data',
                    'recommendation_type': recommendation_type,
                    'available_products': available_products
                })
            
            # Get selected product names for display
            selected_products_info = []
            for stock_code in valid_stock_codes:
                try:
                    product = Dim_Products.objects.get(StockCode=stock_code)
                    selected_products_info.append({
                        'stock_code': stock_code,
                        'description': product.Description
                    })
                except Dim_Products.DoesNotExist:
                    selected_products_info.append({
                        'stock_code': stock_code,
                        'description': 'Unknown Product'
                    })
            
            recommendations = []
            recommendation_source = ""
            error_message = None
            
            if recommendation_type == 'ai' and CREWAI_AVAILABLE and agents is not None:
                # Try AI recommendations first
                try:
                    print(f"🤖 Getting AI recommendations for customer {target_user_id}...")
                    results = agents.run_recommendations(
                        target_user_id=target_user_id,
                        stock_codes=valid_stock_codes,
                        top_n=top_n
                    )
                    
                    # Parse AI results
                    if hasattr(results, 'raw') and results.raw:
                        try:
                            result_text = results.raw
                            if isinstance(result_text, str):
                                start_idx = result_text.find('{')
                                end_idx = result_text.rfind('}') + 1
                                if start_idx != -1 and end_idx != -1:
                                    json_str = result_text[start_idx:end_idx]
                                    parsed_results = json.loads(json_str)
                                    raw_recommendations = parsed_results.get('Top Recommendations', [])
                                    
                                    for rec in raw_recommendations:
                                        recommendations.append({
                                            'stock_code': rec.get('Stock Code', ''),
                                            'description': rec.get('Description', ''),
                                            'unit_price': rec.get('Unit Price', 0),
                                            'confidence': rec.get('Confidence', 0)
                                        })
                                    
                                    recommendation_source = "AI Algorithm (CrewAI) - SQLite Data"
                        except (json.JSONDecodeError, AttributeError) as e:
                            print(f"Error parsing AI results: {e}")
                            raise Exception("Failed to parse AI recommendations")
                
                except Exception as e:
                    print(f"⚠️ AI recommendations failed: {e}")
                    error_message = f"AI system unavailable: {str(e)}"
                    # Fall back to manual recommendations
                    recommendation_type = 'manual'
            
            # Use manual recommendations if AI failed or was requested
            if not recommendations or recommendation_type == 'manual':
                print(f"📊 Getting manual recommendations for customer {target_user_id}...")
                print(f"📊 Valid stock codes: {valid_stock_codes}")
                
                try:
                    # First try the weighted hybrid approach
                    all_recommendations = []
                    
                    for stock_code in valid_stock_codes:
                        print(f"📊 Processing stock code: {stock_code}")
                        
                        # Check if stock code exists in data
                        if stock_code not in df_clean['StockCode'].values:
                            print(f"⚠️ Stock code {stock_code} not found in data")
                            continue
                            
                        hybrid_result = weighted_hybrid_recommendations(
                            input_stock_code=stock_code,
                            target_user_id=target_user_id,
                            data=df_clean,
                            top_n=3
                        )
                        
                        print(f"📊 Hybrid result for {stock_code}: {hybrid_result}")
                        
                        if 'Top Recommendations' in hybrid_result and hybrid_result['Top Recommendations']:
                            print(f"📊 Found {len(hybrid_result['Top Recommendations'])} recommendations for {stock_code}")
                            for rec in hybrid_result['Top Recommendations']:
                                all_recommendations.append({
                                    'stock_code': rec.get('Stock Code', ''),
                                    'description': rec.get('Description', ''),
                                    'unit_price': rec.get('Unit Price', 0),
                                    # 'score': 1.0
                                })
                        else:
                            print(f"📊 No recommendations found for {stock_code}")
                    
                    print(f"📊 Total hybrid recommendations: {len(all_recommendations)}")
                    
                    # If weighted hybrid didn't work, use simple recommendation
                    if not all_recommendations:
                        print("📊 Weighted hybrid failed, trying simple recommendations...")
                        all_recommendations = get_simple_recommendations(df_clean, target_user_id, valid_stock_codes, top_n)
                    
                    # Remove duplicates and limit to top_n
                    seen_stocks = set()
                    for rec in all_recommendations:
                        if rec['stock_code'] not in seen_stocks and len(recommendations) < top_n:
                            recommendations.append(rec)
                            seen_stocks.add(rec['stock_code'])
                            print(f"📊 Added recommendation: {rec['stock_code']} - {rec['description']}")
                    
                    print(f"📊 Final recommendations count: {len(recommendations)}")
                    
                    if recommendation_type == 'ai' and error_message:
                        recommendation_source = "Manual Algorithm (Fallback) - SQLite Data"
                    else:
                        recommendation_source = "Manual Algorithm (Hybrid + Simple) - SQLite Data"
                        
                except Exception as e:
                    error_message = f"Manual recommendations failed: {str(e)}"
                    print(f"❌ Manual recommendations error: {e}")
                    import traceback
                    print(f"❌ Full traceback: {traceback.format_exc()}")
            
            return render(request, 'recommendations/index.html', {
                'recommendations': recommendations,
                'target_user_id': target_user_id,
                'selected_products': selected_products_info,
                'valid_stock_codes': valid_stock_codes,
                'recommendation_type': recommendation_type,
                'recommendation_source': recommendation_source,
                'error': error_message,
                'available_products': available_products
            })
            
        except ValueError:
            return render(request, 'recommendations/index.html', {
                'error': 'Please enter a valid customer ID (number)',
                'recommendation_type': recommendation_type,
                'available_products': available_products
            })
        except Exception as e:
            return render(request, 'recommendations/index.html', {
                'error': f'Error processing request: {str(e)}',
                'recommendation_type': recommendation_type,
                'available_products': available_products
            })
    
    # GET request - show empty form with product dropdown
    return render(request, 'recommendations/index.html', {
        'available_products': available_products
    })