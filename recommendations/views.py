import json
import sys
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.crew_agents import RecommendationAgents
from src.data.processor import DataProcessor

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
        df = _processor.load_dataset()
        if df is not None:
            _df_clean = _processor.clean_data(df)
            print(f"✅ Data loaded: {len(_df_clean)} rows")
        else:
            print("❌ Failed to load dataset")
            return None, None, None
    
    if _agents is None:
        print("🤖 Initializing AI agents...")
        _agents = RecommendationAgents()
        if _df_clean is not None:
            _agents.set_dataframe(_df_clean)
    
    return _processor, _df_clean, _agents

def index(request):
    """Main page with recommendation form."""
    if request.method == 'POST':
        # Get form data
        target_user_id = request.POST.get('target_user_id')
        stock_codes_input = request.POST.get('stock_codes')
        top_n = int(request.POST.get('top_n', 5))
        
        # Parse stock codes
        stock_codes = [code.strip() for code in stock_codes_input.split(',') if code.strip()]
        
        try:
            target_user_id = int(target_user_id)
            
            # Get data and agents
            processor, df_clean, agents = get_processor_and_data()
            if df_clean is None or agents is None:
                return render(request, 'recommendations/index.html', {
                    'error': 'Failed to load data or initialize agents'
                })
            
            # Validate customer exists
            if target_user_id not in df_clean['CustomerID'].unique():
                available_customers = df_clean['CustomerID'].unique()[:10].tolist()
                return render(request, 'recommendations/index.html', {
                    'error': f'Customer {target_user_id} not found',
                    'available_customers': available_customers
                })
            
            # Validate stock codes
            available_stocks = df_clean['StockCode'].unique()
            valid_stock_codes = [code for code in stock_codes if code in available_stocks]
            
            if not valid_stock_codes:
                return render(request, 'recommendations/index.html', {
                    'error': f'None of the stock codes found: {stock_codes}. Please enter valid stock codes.'
                })
            
            # Get recommendations
            print(f"🔍 Getting recommendations for customer {target_user_id}...")
            results = agents.run_recommendations(
                target_user_id=target_user_id,
                stock_codes=valid_stock_codes,
                top_n=top_n
            )
            
            # Parse results
            recommendations = []
            if hasattr(results, 'raw') and results.raw:
                try:
                    # Parse the JSON from the results
                    result_text = results.raw
                    if isinstance(result_text, str):
                        # Find JSON in the text
                        start_idx = result_text.find('{')
                        end_idx = result_text.rfind('}') + 1
                        if start_idx != -1 and end_idx != -1:
                            json_str = result_text[start_idx:end_idx]
                            parsed_results = json.loads(json_str)
                            raw_recommendations = parsed_results.get('Top Recommendations', [])
                            
                            # Convert keys to template-friendly format (fix for Django template syntax)
                            recommendations = []
                            for rec in raw_recommendations:
                                recommendations.append({
                                    'stock_code': rec.get('Stock Code', ''),
                                    'description': rec.get('Description', ''),
                                    'unit_price': rec.get('Unit Price', 0),
                                    # 'source': rec.get('Source', ''),
                                    # 'popularity': rec.get('Popularity', 0)
                                })
                                
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing results: {e}")
                    recommendations = []
            
            return render(request, 'recommendations/index.html', {
                'recommendations': recommendations,
                'target_user_id': target_user_id,
                'stock_codes': valid_stock_codes,
                'top_n': top_n
            })
            
        except ValueError:
            return render(request, 'recommendations/index.html', {
                'error': 'Please enter a valid customer ID (number)'
            })
        except Exception as e:
            return render(request, 'recommendations/index.html', {
                'error': f'Error processing request: {str(e)}'
            })
    
    # GET request - show empty form
    return render(request, 'recommendations/index.html')