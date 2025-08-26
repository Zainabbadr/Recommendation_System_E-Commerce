import json
import sys
from pathlib import Path
from django.shortcuts import render, redirect
from django.http import JsonResponse
# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from src.data.processor import DataProcessor
from src.models.recommendations import weighted_hybrid_recommendations, CollaborativeFiltering
from recommendations.models import Dim_Products  # Import Django model
from recommendations.models import Dim_Customers
from src.chatbot.langgraph_chatbot import RecommendationChatbot
import os

# Imports for plotting
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
# Set Matplotlib backend to 'Agg' for non-interactive plotting
plt.switch_backend('Agg')


# Global variables to cache data
_processor = None
_df_clean = None
_chatbot = None
from recommendations.models import ChatHistory
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def get_chat_history(request):
    """API endpoint to get chat history"""
    try:
        customer_id = request.session.get('customer_id')
        
        # Get chats for this customer or general chats
        if customer_id:
            chats = ChatHistory.objects.filter(
                models.Q(customer_id=customer_id) | models.Q(customer_id__isnull=True)
            ).order_by('-last_updated')[:50]
        else:
            chats = ChatHistory.objects.filter(customer_id__isnull=True).order_by('-last_updated')[:50]
        
        chat_data = []
        for chat in chats:
            messages = chat.get_messages()
            
            chat_data.append({
                'id': chat.chat_id,
                'title': chat.title,
                'messages': messages,
                'lastUpdated': chat.last_updated.isoformat() if chat.last_updated else None,
                'created': chat.created.isoformat() if chat.created else None,
                'messageCount': len(messages)
            })
        
        return JsonResponse({'chats': chat_data, 'status': 'success'})
        
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        return JsonResponse({'error': str(e), 'status': 'error'})

@csrf_exempt
def save_chat_messages(request):
    """API endpoint to save chat messages"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        chat_id = data.get('chat_id')
        title = data.get('title', 'New Chat')
        messages = data.get('messages', [])
        customer_id = request.session.get('customer_id')
        
        if not chat_id:
            return JsonResponse({'error': 'chat_id required'}, status=400)
        
        # Update or create chat
        chat, created = ChatHistory.objects.update_or_create(
            chat_id=chat_id,
            defaults={
                'title': title,
                'customer_id': customer_id,
            }
        )
        
        # Update messages
        chat.set_messages(messages)
        chat.save()
        
        return JsonResponse({
            'status': 'success',
            'created': created,
            'chat_id': chat_id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e), 'status': 'error'})

@csrf_exempt
def delete_chat(request):
    """API endpoint to delete a chat"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        chat_id = data.get('chat_id')
        
        if not chat_id:
            return JsonResponse({'error': 'chat_id required'}, status=400)
        
        deleted_count, _ = ChatHistory.objects.filter(chat_id=chat_id).delete()
        
        return JsonResponse({
            'status': 'success',
            'deleted': deleted_count > 0,
            'chat_id': chat_id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e), 'status': 'error'})
    
@csrf_exempt
def get_purchase_history(request):
    """API endpoint to get customer purchase history"""
    try:
        customer_id = request.session.get('customer_id')
        
        if not customer_id:
            return JsonResponse({
                'error': 'Customer ID not found in session',
                'status': 'error'
            }, status=400)
        
        # Get processor and data
        processor, df_clean = get_processor_and_data()
        
        if df_clean is None:
            return JsonResponse({
                'error': 'Failed to load transaction data',
                'status': 'error'
            }, status=500)
        
        # Filter data for the specific customer
        customer_purchases = df_clean[df_clean['CustomerID'] == customer_id].copy()
        
        if customer_purchases.empty:
            return JsonResponse({
                'purchases': [],
                'total_count': 0,
                'total_spent': 0,
                'status': 'success',
                'message': f'No purchase history found for Customer ID {customer_id}'
            })
        
        # Calculate total price if not exists
        if 'TotalPrice' not in customer_purchases.columns:
            customer_purchases['TotalPrice'] = customer_purchases['Quantity'] * customer_purchases['UnitPrice']
        
        # Sort by date (most recent first)
        customer_purchases['InvoiceDate'] = pd.to_datetime(customer_purchases['InvoiceDate'])
        customer_purchases = customer_purchases.sort_values('InvoiceDate', ascending=False)
        
        # Prepare purchase data
        purchases = []
        for _, row in customer_purchases.iterrows():
            purchases.append({
                'invoice_no': row['InvoiceNo'],
                'stock_code': row['StockCode'],
                'description': row['Description'] if pd.notna(row['Description']) else 'No Description',
                'quantity': int(row['Quantity']),
                'unit_price': float(row['UnitPrice']),
                'total_price': float(row['TotalPrice']),
                'purchase_date': row['InvoiceDate'].strftime('%Y-%m-%d'),
                'purchase_time': row['InvoiceDate'].strftime('%H:%M:%S'),
                'purchase_datetime': row['InvoiceDate'].strftime('%b %d, %Y at %I:%M %p'),
                'country': row.get('Country', 'Unknown')
            })
        
        # Calculate summary statistics
        total_spent = float(customer_purchases['TotalPrice'].sum())
        unique_products = customer_purchases['StockCode'].nunique()
        total_orders = customer_purchases['InvoiceNo'].nunique()
        avg_order_value = total_spent / total_orders if total_orders > 0 else 0
        
        return JsonResponse({
            'purchases': purchases,
            'total_count': len(purchases),
            'total_spent': total_spent,
            'unique_products': unique_products,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in get_purchase_history: {e}")
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)
    

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
    """Handle chatbot API requests with chat history integration."""
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        user_id = data.get('user_id', 'default_user')
        chat_id = data.get('chat_id')
        session_customer_id = request.session.get('customer_id')
        
        if not message.strip():
            return JsonResponse({
                'error': 'Message cannot be empty',
                'status': 'error'
            }, status=400)
        
        # Generate chat_id if not provided
        if not chat_id:
            import time
            chat_id = f"chat_{int(time.time())}"
        
        # Get chatbot instance
        chatbot = get_chatbot()
        
        if chatbot is None:
            return JsonResponse({
                'response': 'Sorry, the AI chatbot is currently unavailable. Please try the AI Recommendations tab.',
                'status': 'fallback'
            })
        
        # Get chatbot response (now with Django integration)
        response = chatbot.chat(message, chat_id=chat_id, user_id=user_id, session_customer_id=session_customer_id)
        
        return JsonResponse({
            'response': response,
            'status': 'success',
            'chat_id': chat_id,
            'user_id': user_id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data',
            'status': 'error'
        }, status=400)
    except Exception as e:
        print(f"Chatbot error: {e}")
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

def login_view(request):
    """Handle login requests and validate customer IDs."""
    if request.method == 'POST':
        customer_id = request.POST.get('customer_id')
        
        if not customer_id:
            return render(request, 'recommendations/login.html', {
                'error': 'Please enter your Customer ID'
            })
        
        try:
            customer_id = int(customer_id)
            
            # Validate customer exists in database
            from recommendations.models import Dim_Customers
            
            try:
                customer = Dim_Customers.objects.get(CustomerID=customer_id)
                # Store customer ID in session
                request.session['customer_id'] = customer_id
                return redirect('recommendations:index')
            except Dim_Customers.DoesNotExist:
                return render(request, 'recommendations/login.html', {
                    'error': 'Customer ID not found in database'
                })
                
        except ValueError:
            return render(request, 'recommendations/login.html', {
                'error': 'Please enter a valid Customer ID (number)'
            })
    
    return render(request, 'recommendations/login.html')

def logout_view(request):
    """Handle logout requests."""
    # Clear the session
    if 'customer_id' in request.session:
        del request.session['customer_id']
    return redirect('recommendations:login')

def index(request):
    """Main page with dual-tab recommendation form."""
    # Check if customer is logged in
    customer_id = request.session.get('customer_id')
    
    # Get products for dropdown (always needed for GET and POST)
    available_products = get_products_for_dropdown()

    # Handle purchase history tab request
    if request.method == 'GET' and request.GET.get('tab') == 'purchase-history':
        if not customer_id:
            return render(request, 'recommendations/index.html', {
                'error': 'Please login to view your purchase history',
                'available_products': available_products,
                'active_tab': 'purchase-history'
            })
    
    if request.method == 'POST':
        # Get form data
        target_user_id = request.POST.get('target_user_id')
        selected_stock_codes = request.POST.getlist('selected_products')
        top_n = 7  # Always 7 recommendations
        
        # Use session customer ID if not provided
        if not target_user_id and customer_id:
            target_user_id = str(customer_id)
        
        try:
            target_user_id = int(target_user_id)
            
            # Validate that products were selected
            if not selected_stock_codes:
                return render(request, 'recommendations/index.html', {
                    'error': 'Please select at least one product',
                    'available_products': available_products,
                    'customer_id': customer_id
                })
            
            # Get data
            processor, df_clean = get_processor_and_data()
            if df_clean is None:
                return render(request, 'recommendations/index.html', {
                    'error': 'Failed to load data',
                    'available_products': available_products,
                    'customer_id': customer_id
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
        'available_products': available_products,
        'customer_id': customer_id, # Pass customer_id to index.html for dashboard link
        'active_tab': request.GET.get('tab', 'chatbot')  
    })


def customer_dashboard_view(request):
    """
    Displays a modern dashboard for a specific customer, including purchase history,
    top products, and customer information with updated styling.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    import seaborn as sns
    import numpy as np
    
    customer_id = request.session.get('customer_id')

    if not customer_id:
        return redirect('recommendations:login') 

    processor, df_clean = get_processor_and_data()

    if df_clean is None:
        return render(request, 'recommendations/customer_dashboard.html', {
            'error': 'Failed to load data for dashboard.',
            'customer_id': customer_id
        })

    # Ensure 'CustomerID' column is of the same type (int) for comparison
    customer_df = df_clean[df_clean["CustomerID"] == customer_id].copy() 

    if customer_df.empty:
        return render(request, 'recommendations/customer_dashboard.html', {
            'error': f'Customer ID {customer_id} not found in transaction data.',
            'customer_id': customer_id
        })

    # Ensure 'TotalPrice' column exists or is derived
    if 'TotalPrice' not in customer_df.columns:
        customer_df['TotalPrice'] = customer_df['Quantity'] * customer_df['UnitPrice']

    # Placeholder for 'segment' if not already in df_clean
    if 'segment' not in customer_df.columns:
        customer_df['segment'] = 'Regular'

    # Calculate segment and customer metrics for context
    customer_obj = Dim_Customers.objects.get(CustomerID=customer_id)
    segment = customer_obj.Segment
    total_spend = customer_df["TotalPrice"].sum() if not customer_df.empty else 0.0
    total_orders = customer_df["InvoiceNo"].nunique() if not customer_df.empty else 0
    avg_order_value = total_spend / total_orders if total_orders > 0 else 0
    first_purchase = customer_df["InvoiceDate"].min().strftime("%b %d, %Y") if not customer_df["InvoiceDate"].empty else 'N/A'
    last_purchase = customer_df["InvoiceDate"].max().strftime("%b %d, %Y") if not customer_df["InvoiceDate"].empty else 'N/A'
    unique_products = customer_df["Description"].nunique() if not customer_df.empty else 0

    # Modern color palette matching UI
    colors = {
        'primary': '#667eea',      # Purple-blue gradient start
        'secondary': '#764ba2',    # Purple gradient end
        'accent': '#ff6b35',       # Modern orange
        'background': '#f8f9fa',   # Light background
        'text': '#2c3e50',         # Dark text
        'grid': '#e9ecef',         # Light grid
        'success': '#28a745',      # Green
        'gradient_colors': ['#667eea', '#7c6eea', '#926eea', '#a76eea', '#bd6eea', '#d36eea', '#e96eea']
    }

    # Generate modern plots
    plot_data = {}
    try:
        # Set modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Try to use Inter font (fallback to system fonts if not available)
        try:
            plt.rcParams['font.family'] = [ 'Arial', 'DejaVu Sans']
        except:
            plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
            
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        # Remove top and right spines for cleaner look
        for ax in axs.flat:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(colors['grid'])
            ax.spines['bottom'].set_color(colors['grid'])
            ax.tick_params(colors=colors['text'], which='both')
            ax.grid(True, alpha=0.3, color=colors['grid'])

        # 1️⃣ Purchases Over Time - Modern Line Chart
        customer_df["InvoiceDate"] = pd.to_datetime(customer_df["InvoiceDate"])
        if not customer_df.empty:
            # Create daily spending data
            daily_spending = customer_df.set_index("InvoiceDate").resample('D')["TotalPrice"].sum()
            daily_spending = daily_spending[daily_spending > 0]  # Remove zero days
            
            # Plot with gradient-like effect
            line = axs[0,0].plot(daily_spending.index, daily_spending.values, 
                               color=colors['primary'], linewidth=3, alpha=0.8, 
                               marker='o', markersize=6, markerfacecolor=colors['accent'],
                               markeredgecolor='white', markeredgewidth=2)
            
            # Fill area under curve with gradient effect
            axs[0,0].fill_between(daily_spending.index, daily_spending.values, 
                                alpha=0.2, color=colors['primary'])
            
        axs[0,0].set_title("Purchase Trends Over Time", fontweight='bold', 
                          color=colors['text'], pad=20, fontsize=14)
        # axs[0,0].set_xlabel("Date", color=colors['text'], fontweight='600')
        axs[0,0].set_ylabel("Daily Spending ($)", color=colors['text'], fontweight='600')
        axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axs[0,0].tick_params(axis='x', rotation=45)

        # 2️⃣ Top Products - Modern Horizontal Bar Chart
        if not customer_df.empty:
            top_products = customer_df.groupby("Description")["Quantity"].sum().sort_values(ascending=True).tail(7)
            
            # Create gradient colors
            n_bars = len(top_products)
            gradient_colors = [colors['primary'] if i % 2 == 0 else colors['accent'] 
                             for i in range(n_bars)]
            
            bars = axs[0,1].barh(range(len(top_products)), top_products.values, 
                               color=gradient_colors, alpha=0.8, height=0.6)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_products.values)):
                axs[0,1].text(value + max(top_products.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{int(value)}', va='center', ha='left', fontweight='600', 
                            color=colors['text'], fontsize=9)
            
            axs[0,1].set_yticks(range(len(top_products)))
            axs[0,1].set_yticklabels([desc[:25] + '...' if len(desc) > 25 else desc 
                                    for desc in top_products.index], fontsize=9)
            
        axs[0,1].set_title("Most Purchased Products", fontweight='bold', 
                          color=colors['text'], pad=20, fontsize=14)
        axs[0,1].set_xlabel("Quantity Purchased", color=colors['text'], fontweight='600')

        # 3️⃣ Monthly Purchase Frequency - 3D-like Pie Chart with Quantities
        if not customer_df.empty:
            monthly_purchases = customer_df.groupby(customer_df["InvoiceDate"].dt.to_period('M')).size()

            if len(monthly_purchases) > 0:
                # Alternate colors: primary / secondary
                pie_colors = [colors['primary'] if i % 2 == 0 else colors['secondary'] 
                            for i in range(len(monthly_purchases))]

                # Show actual values instead of %
                def make_autopct(values):
                    def my_autopct(pct):
                        total = sum(values)
                        val = int(round(pct*total/100.0))
                        return f"{val}"   # show raw value
                    return my_autopct

                wedges, texts, autotexts = axs[1,0].pie(
                    monthly_purchases.values,
                    labels=[str(period) for period in monthly_purchases.index],
                    autopct=make_autopct(monthly_purchases.values),
                    startangle=140,
                    colors=pie_colors,
                    shadow=True,       # gives "3D shadow" effect
                    explode=[0.05]*len(monthly_purchases),  # pop-out for 3D look
                    textprops={'color': colors['text'], 'fontsize': 9, 'fontweight': '600'}
                )

                # Style quantity labels inside pie
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontweight("bold")
                    autotext.set_fontsize(9)

        axs[1,0].set_title("Monthly Purchase Frequency", fontweight='bold', 
                        color=colors['text'], pad=20, fontsize=14)



        # 4️⃣ Customer Info Card - Modern Table Design (UPDATED: Table format)
        axs[1,1].axis("off")
        
        # Create table data
        table_data = [
            ['Segment', segment],
            ['Total Spending', f'${total_spend:,.2f}'],
            ['Total Orders', f'{total_orders:,}'],
            ['Avg Order Value', f'${avg_order_value:.2f}'],
            ['Unique Products', f'{unique_products:,}'],
            ['First Purchase', first_purchase],
            ['Latest Purchase', last_purchase]
        ]
        
        # Create table
        table = axs[1,1].table(cellText=table_data,
                              colLabels=['Metric', 'Value'],
                              cellLoc='left',
                              loc='center',
                              colWidths=[0.4, 0.6])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor(colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.08)
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor(colors['background'])
                else:
                    table[(i, j)].set_facecolor('white')
                table[(i, j)].set_text_props(color=colors['text'])
                table[(i, j)].set_height(0.06)
        
        # Remove table borders for cleaner look
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor(colors['grid'])

        axs[1,1].set_title("Customer Overview", fontweight='bold', 
                          color=colors['text'], pad=20, fontsize=14)

        # Adjust layout with better spacing
        plt.tight_layout(pad=3.0)
        
        # Adjust subplot spacing for better visual balance
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Save plot with high quality
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150, 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_data['dashboard_image'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

    except Exception as e:
        print(f"Error generating modern dashboard plots for customer {customer_id}: {e}")
        plot_data['error'] = f"Error generating dashboard: {e}"

    # Return context with segment for header replacement
    return render(request, 'recommendations/customer_dashboard.html', {
        'customer_id': customer_id,
        'customer_segment': segment,  # Added for template usage
        'dashboard_image': plot_data.get('dashboard_image'),
        'error': plot_data.get('error')
    })

