import json
import sys
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from src.data.processor import DataProcessor
from src.models.recommendations import weighted_hybrid_recommendations, CollaborativeFiltering
from recommendations.models import Dim_Products  # Import Django model
from src.chatbot.langgraph_chatbot import RecommendationChatbot
import os
# from django.http import StreamingHttpResponse

# Global variables to cache data
_processor = None
_df_clean = None
_chatbot = None

def get_processor_and_data():
    """Initialize processor and load data if not already done."""
    global _processor, _df_clean
    
    if _processor is None:
        print("🔧 Initializing data processor...")
        _processor = DataProcessor()
        df = _processor.load_data_from_sqlite()
        if df is not None:
            _df_clean = _processor.clean_data(df)
            print(f"✅ Data loaded: {len(_df_clean)} rows")
        else:
            print("❌ Failed to load dataset")
            return None, None
    
    return _processor, _df_clean

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
            })
        
        print(f"✅ Generated {len(recommendations)} simple recommendations")
        return recommendations
        
    except Exception as e:
        print(f"❌ Simple recommendation error: {e}")
        return []

@csrf_exempt
@require_http_methods(["POST"])
def chatbot_view(request):
    """Handle chatbot API requests."""
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
                'response': 'Sorry, the AI chatbot is currently unavailable. Please try the AI Recommendations tab.',
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
            'response': 'I encountered an unexpected error. Please try again or use the AI Recommendations tab.',
            'status': 'error'
        })

def health_check(request):
    """Health check endpoint."""
    try:
        chatbot = get_chatbot()
        return JsonResponse({
            'status': 'healthy',
            'service': 'django',
            'chatbot_ready': chatbot is not None
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        })

def index(request):
    """Main page with dual-tab recommendation form."""
    # Get products for dropdown (always needed for GET and POST)
    available_products = get_products_for_dropdown()
    
    if request.method == 'POST':
        # Get form data
        target_user_id = request.POST.get('target_user_id')
        selected_stock_codes = request.POST.getlist('selected_products')
        top_n = 7  # Always 7 recommendations
        
        try:
            target_user_id = int(target_user_id)
            
            # Validate that products were selected
            if not selected_stock_codes:
                return render(request, 'recommendations/index.html', {
                    'error': 'Please select at least one product',
                    'available_products': available_products
                })
            
            # Get data
            processor, df_clean = get_processor_and_data()
            if df_clean is None:
                return render(request, 'recommendations/index.html', {
                    'error': 'Failed to load data',
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
                    'available_products': available_products
                })
            
            # Validate selected stock codes exist in data
            available_stocks = df_clean['StockCode'].unique()
            valid_stock_codes = [code for code in selected_stock_codes if code in available_stocks]
            
            if not valid_stock_codes:
                return render(request, 'recommendations/index.html', {
                    'error': f'Selected products not found in transaction data',
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
            
            # AI Recommendations using LangGraph with Manual Fallback
            print(f"🚀 Getting LangGraph AI recommendations for customer {target_user_id}...")
            
            try:
                # Try LangGraph first
                chatbot = get_chatbot()
                
                if chatbot is None:
                    raise Exception("LangGraph chatbot not available")
                
                # Use LangGraph recommendation pipeline directly
                print(f"📊 Running LangGraph collaborative filtering...")
                collaborative_results = chatbot._collaborative_filtering(target_user_id, valid_stock_codes, top_n=10)
                
                print(f"📊 Running LangGraph content-based filtering...")
                content_based_results = chatbot._content_based_filtering(target_user_id, valid_stock_codes, recommendations_per_stock=3)
                
                print(f"📊 Running LangGraph reranking...")
                final_recommendations = chatbot._rerank_recommendations(collaborative_results, content_based_results, top_n=7)
                
                # Convert to expected format
                if final_recommendations:
                    for rec in final_recommendations:
                        recommendations.append({
                            'stock_code': rec.get('Stock Code', ''),
                            'description': rec.get('Description', ''),
                            'unit_price': rec.get('Unit Price', 0),
                            'source': rec.get('Source', 'LangGraph AI'),
                            'popularity': rec.get('Popularity', 0)
                        })
                    
                    recommendation_source = "🚀 LangGraph AI Pipeline (Collaborative + Content-Based + Reranking)"
                    print(f"✅ LangGraph generated {len(recommendations)} recommendations")
                else:
                    raise Exception("LangGraph returned no recommendations")
                    
            except Exception as e:
                print(f"⚠️ LangGraph AI failed: {e}. Falling back to manual algorithms...")
                error_message = f"AI temporarily unavailable, using backup algorithms"
                
                # **AUTOMATIC FALLBACK TO MANUAL (HIDDEN FROM USER)**
                try:
                    print(f"📊 Getting manual fallback recommendations for customer {target_user_id}...")
                    
                    # Use manual algorithms
                    all_recommendations = []
                    
                    for stock_code in valid_stock_codes:
                        print(f"📊 Processing stock code: {stock_code}")
                        
                        if stock_code not in df_clean['StockCode'].values:
                            print(f"⚠️ Stock code {stock_code} not found in data")
                            continue
                            
                        # Try weighted hybrid approach
                        hybrid_result = weighted_hybrid_recommendations(
                            input_stock_code=stock_code,
                            target_user_id=target_user_id,
                            data=df_clean,
                            top_n=3
                        )
                        
                        if 'Top Recommendations' in hybrid_result and hybrid_result['Top Recommendations']:
                            for rec in hybrid_result['Top Recommendations']:
                                all_recommendations.append({
                                    'stock_code': rec.get('Stock Code', ''),
                                    'description': rec.get('Description', ''),
                                    'unit_price': rec.get('Unit Price', 0),
                                    'source': 'Manual Fallback',
                                    'popularity': 0
                                })
                    
                    # If weighted hybrid didn't work, use simple recommendation
                    if not all_recommendations:
                        print("📊 Weighted hybrid failed, trying simple recommendations...")
                        simple_recs = get_simple_recommendations(df_clean, target_user_id, valid_stock_codes, top_n)
                        for rec in simple_recs:
                            all_recommendations.append({
                                'stock_code': rec['stock_code'],
                                'description': rec['description'],
                                'unit_price': rec['unit_price'],
                                'source': 'Simple Fallback',
                                'popularity': 0
                            })
                    
                    # Remove duplicates and limit to top_n
                    seen_stocks = set()
                    for rec in all_recommendations:
                        if rec['stock_code'] not in seen_stocks and len(recommendations) < top_n:
                            recommendations.append(rec)
                            seen_stocks.add(rec['stock_code'])
                    
                    recommendation_source = "🔄 Manual Algorithms (Automatic Fallback) - Hybrid + Simple"
                    print(f"✅ Manual fallback generated {len(recommendations)} recommendations")
                    
                except Exception as fallback_error:
                    print(f"❌ Manual fallback also failed: {fallback_error}")
                    error_message = f"Both AI and manual systems failed: {str(fallback_error)}"
                    recommendation_source = "❌ System Error"
            
            return render(request, 'recommendations/index.html', {
                'recommendations': recommendations,
                'target_user_id': target_user_id,
                'selected_products': selected_products_info,
                'valid_stock_codes': valid_stock_codes,
                'recommendation_source': recommendation_source,
                'error': error_message,
                'available_products': available_products
            })
            
        except ValueError:
            return render(request, 'recommendations/index.html', {
                'error': 'Please enter a valid customer ID (number)',
                'available_products': available_products
            })
        except Exception as e:
            return render(request, 'recommendations/index.html', {
                'error': f'Error processing request: {str(e)}',
                'available_products': available_products
            })
    
    # GET request - show empty form with product dropdown
    return render(request, 'recommendations/index.html', {
        'available_products': available_products
    })