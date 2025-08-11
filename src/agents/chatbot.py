"""
E-commerce Conversational AI Chatbot
Integrates with CrewAI recommendation system for personalized product suggestions
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from crew_agents import RecommendationAgents  # CHANGED: relative import
# For the chatbot LLM - using Groq directly
import groq  # CHANGED: using groq directly instead of CrewAI LLM

class IntentType(Enum):
    """Different types of user intents the chatbot can handle"""
    GENERAL_RECOMMENDATION = "general_recommendation"
    SIMILAR_TO_ITEM = "similar_to_item"
    PURCHASE_HISTORY = "purchase_history"
    EVENT_BASED = "event_based"
    BUDGET_BASED = "budget_based"
    CATEGORY_BASED = "category_based"
    COMPARISON = "comparison"
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"

@dataclass
class UserContext:
    """Stores user conversation context"""
    customer_id: Optional[int] = None
    last_recommendations: List[Dict] = None
    conversation_history: List[Dict] = None
    current_intent: Optional[IntentType] = None
    extracted_entities: Dict = None
    last_query_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_recommendations is None:
            self.last_recommendations = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.extracted_entities is None:
            self.extracted_entities = {}

class EcommerceConversationalBot:
    """
    Conversational AI chatbot for e-commerce recommendations
    Integrates with CrewAI recommendation system
    """
    
    def __init__(self, recommendation_system, df: pd.DataFrame, llm_config: Dict = None):
        """
        Initialize the conversational bot
        
        Args:
            recommendation_system: Your CrewAI RecommendationAgents instance
            df: Your e-commerce dataframe with CustomerID, StockCode, Description, etc.
            llm_config: Configuration for the conversational LLM
        """
        self.recommendation_system = recommendation_system
        self.df = df
        self.user_contexts: Dict[int, UserContext] = {}
        
        # CHANGED: Set up Groq client directly instead of CrewAI LLM
        try:
            self.groq_client = groq.Groq(
                api_key=os.getenv("GROQ_API_KEY")  # Replace with your actual key
            )
            print("✅ Groq client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Groq client: {e}")
            self.groq_client = None
        
        # Set dataframe for recommendation system
        self.recommendation_system.set_dataframe(df)
        
        # Initialize intent patterns
        self._setup_intent_patterns()
        
        # Bot personality and context
        self.bot_personality = """
        You are Alex, a friendly and knowledgeable e-commerce shopping assistant. 
        You help customers find products they'll love based on their preferences and purchase history.
        You're conversational, helpful, and always try to understand what the customer really wants.
        Keep responses concise but warm and personal.
        """

    def _setup_intent_patterns(self):
        """Set up regex patterns for intent recognition"""
        self.intent_patterns = {
            IntentType.GENERAL_RECOMMENDATION: [
                r"recommend.*(?:something|anything|products)",
                r"what should i (?:buy|get|purchase)",
                r"show me (?:some|new) (?:products|items)",
                r"i need (?:something|help finding)",
                r"suggestions?",
            ],
            IntentType.SIMILAR_TO_ITEM: [
                r"similar to.*",
                r"like (?:the|this) (?:.*i bought|.*i purchased|.*i got)",
                r"more (?:products|items) like",
                r"(?:i liked|i loved).*(?:recommend|suggest|show).*(?:similar|like)",
                r"alternatives? (?:to|for)",
            ],
            IntentType.PURCHASE_HISTORY: [
                r"what (?:did i|have i) (?:buy|bought|purchase|purchased)",
                r"(?:my|show my) (?:order|purchase|shopping) history",
                r"what's in my (?:cart|orders|purchases)",
                r"items? i (?:bought|purchased|ordered)",
            ],
            IntentType.BUDGET_BASED: [
                r"under \$?\d+",
                r"cheaper (?:than|alternatives?)",
                r"budget.*\$?\d+",
                r"(?:affordable|inexpensive|cheap) (?:options?|alternatives?)",
                r"less than \$?\d+",
            ],
            IntentType.EVENT_BASED: [
                r"(?:for|gift for) (?:birthday|christmas|anniversary|wedding)",
                r"(?:holiday|special occasion) (?:gift|shopping)",
                r"present for",
            ],
            IntentType.COMPARISON: [
                r"compare.*(?:with|to|vs)",
                r"difference between",
                r"which is better",
                r"(?:pros and cons|advantages)",
            ],
            IntentType.GREETING: [
                r"^(?:hi|hello|hey|good (?:morning|afternoon|evening))",
                r"^(?:thanks|thank you)",
                r"^(?:bye|goodbye|see you)",
            ],
            IntentType.HELP: [
                r"help",
                r"how (?:do|can) (?:i|you)",
                r"what can you do",
                r"commands?",
            ],
        }

    def _extract_entities(self, message: str, context: UserContext) -> Dict:
        """Extract entities from user message"""
        entities = {}
        
        # Extract price ranges
        price_match = re.search(r'\$?(\d+(?:\.\d{2})?)', message.lower())
        if price_match:
            entities['price_limit'] = float(price_match.group(1))
            
        # Extract product references from purchase history
        if context.customer_id:
            user_items = self.df[self.df['CustomerID'] == context.customer_id]
            for _, item in user_items.iterrows():
                desc_words = item['Description'].lower().split()
                if any(word in message.lower() for word in desc_words[:3]):  # Match first 3 words
                    entities['referenced_item'] = {
                        'stock_code': item['StockCode'],
                        'description': item['Description'],
                        'price': item['UnitPrice']
                    }
                    break
        
        # Extract quantity or count
        quantity_match = re.search(r'(\d+) (?:items?|products?|things?)', message.lower())
        if quantity_match:
            entities['quantity'] = int(quantity_match.group(1))
        else:
            entities['quantity'] = 5  # default
            
        return entities

    def _classify_intent(self, message: str) -> IntentType:
        """Classify user intent from message"""
        message_lower = message.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
                    
        return IntentType.UNKNOWN

    def _get_user_purchase_history(self, customer_id: int, limit: int = 10) -> List[Dict]:
        """Get user's recent purchase history"""
        if customer_id not in self.df['CustomerID'].values:
            return []
            
        user_data = self.df[self.df['CustomerID'] == customer_id]
        recent_items = user_data.drop_duplicates('StockCode').tail(limit)
        
        return [
            {
                'stock_code': row['StockCode'],
                'description': row['Description'],
                'unit_price': row['UnitPrice'],
                'quantity': row['Quantity'] if 'Quantity' in row else 1
            }
            for _, row in recent_items.iterrows()
        ]

    def _format_recommendations(self, recommendations: Any, intent: IntentType) -> str:
        """Format CrewAI recommendations for conversational response"""
        try:
            # Handle different CrewAI output formats
            if hasattr(recommendations, 'raw'):
                result_data = recommendations.raw
            elif isinstance(recommendations, str):
                # Try to parse JSON from string
                result_data = json.loads(recommendations) if recommendations.startswith('{') else recommendations
            else:
                result_data = recommendations

            # Extract recommendations from different possible formats
            recs = []
            if isinstance(result_data, dict):
                if 'Top Recommendations' in result_data:
                    recs = result_data['Top Recommendations']
                elif 'recommendations' in result_data:
                    recs = result_data['recommendations']
            elif isinstance(result_data, list):
                recs = result_data

            if not recs:
                return "I couldn't find any specific recommendations right now. Let me know what you're looking for!"

            # Format based on intent
            if intent == IntentType.SIMILAR_TO_ITEM:
                response = "Based on your preferences, here are some similar items you might like:\n\n"
            elif intent == IntentType.BUDGET_BASED:
                response = "Here are some great options within your budget:\n\n"
            else:
                response = "I found some perfect recommendations for you:\n\n"

            for i, rec in enumerate(recs[:5], 1):
                stock_code = rec.get('Stock Code', rec.get('stock_code', 'N/A'))
                description = rec.get('Description', rec.get('description', 'No description'))
                price = rec.get('Unit Price', rec.get('unit_price', rec.get('price', 0)))
                
                response += f"{i}. **{description}**\n"
                response += f"   💰 ${price:.2f} | Code: {stock_code}\n\n"

            response += "Would you like more details about any of these items, or should I find something else? 😊"
            return response

        except Exception as e:
            logging.error(f"Error formatting recommendations: {e}")
            return "I found some great products for you, but had trouble formatting them. Could you try asking again?"

    def _generate_conversational_response(self, message: str, context: UserContext, 
                                        recommendations: Optional[Any] = None) -> str:
        """Generate natural conversational response using Groq"""
        
        # CHANGED: Complete replacement of this method
        # Check if Groq client is available
        if not self.groq_client:
            return "I'm here to help you find great products! What are you looking for today?"
        
        # Prepare context for LLM
        conversation_context = ""
        if context.conversation_history:
            recent_history = context.conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                conversation_context += f"User: {exchange['user']}\nBot: {exchange['bot']}\n"

        # Create prompt for conversational LLM
        prompt = f"""
{self.bot_personality}

Conversation History:
{conversation_context}

Current User Message: "{message}"
Intent: {context.current_intent.value if context.current_intent else 'unknown'}

Customer Context:
- Customer ID: {context.customer_id}
- Last Query: {context.last_query_timestamp.strftime('%Y-%m-%d %H:%M') if context.last_query_timestamp else 'First interaction'}

{"Product Recommendations Available: " + str(len(context.last_recommendations)) + " items" if context.last_recommendations else "No recent recommendations"}

Instructions:
1. Respond naturally and conversationally as Alex, the shopping assistant
2. Be helpful, friendly, and personalized
3. If recommendations are available, incorporate them naturally
4. Ask follow-up questions to better understand user needs
5. Keep response concise (2-3 sentences max unless listing products)
6. Use emojis sparingly and appropriately

Generate a helpful, conversational response:
"""

        try:
            # Use Groq client directly
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=512,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error generating conversational response: {e}")
            # Fallback response
            return "I'm here to help you find great products! What are you looking for today?"

    # Rest of your methods remain the same...
    def handle_user_message(self, message: str, customer_id: Optional[int] = None) -> str:
        """
        Main method to handle user messages and return conversational responses
        
        Args:
            message: User's message
            customer_id: Customer ID if known (can be None for guest users)
            
        Returns:
            Conversational response string
        """
        
        # Get or create user context
        if customer_id and customer_id not in self.user_contexts:
            self.user_contexts[customer_id] = UserContext(customer_id=customer_id)
        
        context = self.user_contexts.get(customer_id, UserContext())
        context.last_query_timestamp = datetime.now()
        
        # Classify intent and extract entities
        intent = self._classify_intent(message)
        entities = self._extract_entities(message, context)
        
        context.current_intent = intent
        context.extracted_entities = entities
        
        # Handle different intents
        try:
            if intent == IntentType.GREETING:
                response = self._generate_conversational_response(message, context)
                
            elif intent == IntentType.HELP:
                response = """Hi! I'm Alex, your shopping assistant! 🛍️ I can help you:
                
• Get personalized product recommendations
• Find items similar to what you've bought
• Show your purchase history  
• Find products within your budget
• Compare different products

Just tell me what you're looking for in natural language! For example:
"I loved the blue mug I bought, show me similar items"
"What did I buy last month?"
"Recommend something under $50"
"""

            elif intent == IntentType.PURCHASE_HISTORY:
                if not customer_id:
                    response = "I'd love to show your purchase history! Could you please log in or provide your customer ID?"
                else:
                    history = self._get_user_purchase_history(customer_id)
                    if history:
                        response = "Here's what you've purchased recently:\n\n"
                        for i, item in enumerate(history[-5:], 1):
                            response += f"{i}. {item['description']} - ${item['unit_price']:.2f}\n"
                        response += "\nWould you like recommendations based on any of these items?"
                    else:
                        response = "I don't see any purchase history for your account yet. Ready to find some great products?"

            elif intent in [IntentType.GENERAL_RECOMMENDATION, IntentType.SIMILAR_TO_ITEM, 
                          IntentType.BUDGET_BASED, IntentType.EVENT_BASED]:
                
                if not customer_id:
                    response = "I'd love to give you personalized recommendations! Could you please log in or provide your customer ID?"
                else:
                    # Get user's purchase history for recommendations
                    user_history = self._get_user_purchase_history(customer_id)
                    if not user_history:
                        response = "I don't see any purchase history to base recommendations on. Could you tell me what types of products you're interested in?"
                    else:
                        # Use CrewAI recommendation system
                        stock_codes = [item['stock_code'] for item in user_history[-5:]]  # Last 5 items
                        top_n = entities.get('quantity', 5)
                        
                        # Call your existing CrewAI system
                        recommendations = self.recommendation_system.run_recommendations(
                            target_user_id=customer_id,
                            stock_codes=stock_codes,
                            top_n=top_n
                        )
                        
                        # Store recommendations in context
                        context.last_recommendations = recommendations
                        
                        # Format recommendations for conversation
                        response = self._format_recommendations(recommendations, intent)

            else:  # UNKNOWN or other intents
                response = self._generate_conversational_response(message, context)

            # Store conversation in context
            context.conversation_history.append({
                'user': message,
                'bot': response,
                'timestamp': datetime.now(),
                'intent': intent.value
            })
            
            # Keep only last 10 exchanges
            if len(context.conversation_history) > 10:
                context.conversation_history = context.conversation_history[-10:]
                
            return response
            
        except Exception as e:
            logging.error(f"Error handling user message: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Could you try asking in a different way?"

    def start_chat_session(self, customer_id: Optional[int] = None) -> str:
        """Start a new chat session"""
        if customer_id:
            user_history = self._get_user_purchase_history(customer_id, limit=3)
            if user_history:
                recent_items = ", ".join([item['description'][:30] + "..." if len(item['description']) > 30 
                                        else item['description'] for item in user_history[-2:]])
                return f"Hi! Welcome back! 👋 I see you recently bought {recent_items}. How can I help you find more great products today?"
            else:
                return "Hi there! 🛍️ I'm Alex, your personal shopping assistant. I'm here to help you discover amazing products! What are you looking for today?"
        else:
            return "Hello! 👋 I'm Alex, your shopping assistant. I can help you find great products! For personalized recommendations, please log in with your customer ID. What can I help you with?"

# Example usage and integration
class ChatbotAPI:
    """Simple API wrapper for the chatbot"""
    
    def __init__(self, recommendation_system, df: pd.DataFrame):
        self.bot = EcommerceConversationalBot(recommendation_system, df)
        self.active_sessions: Dict[str, int] = {}  # session_id -> customer_id
    
    def start_session(self, session_id: str, customer_id: Optional[int] = None) -> Dict:
        """Start a new chat session"""
        if customer_id:
            self.active_sessions[session_id] = customer_id
        
        welcome_msg = self.bot.start_chat_session(customer_id)
        return {
            'session_id': session_id,
            'message': welcome_msg,
            'customer_id': customer_id
        }
    
    def send_message(self, session_id: str, message: str) -> Dict:
        """Send a message and get response"""
        customer_id = self.active_sessions.get(session_id)
        response = self.bot.handle_user_message(message, customer_id)
        
        return {
            'session_id': session_id,
            'user_message': message,
            'bot_response': response,
            'timestamp': datetime.now().isoformat()
        }

# Load your dataframe (replace with your actual data loading)
import pandas as pd
import numpy as np

def load_ecommerce_data():
    """
    Load and prepare the 3 CSV files for the chatbot system
    Returns a merged dataframe suitable for the recommendation system
    """
    
    try:
        # Load dimension tables
        dim_customers = pd.read_csv('Dim_Customers.csv')
        dim_products = pd.read_csv('Dim_Products.csv')
        
        # Load fact table (transactions)
        fact_transactions = pd.read_csv('Fact_Transactions.csv')
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None
    
    # Merge transactions with products
    df_with_products = fact_transactions.merge(
        dim_products, 
        on='StockCode',  # Adjust column name as needed
        how='left'
    )
    
    # Merge with customers
    final_df = df_with_products.merge(
        dim_customers,
        on='CustomerID',  # Adjust column name as needed
        how='left'
    )
    
    # The chatbot expects these specific columns - let's map them
    # You may need to adjust these column mappings based on your actual column names
    
    expected_columns = {
        # Map your actual column names to expected names
        'CustomerID': 'CustomerID',        # Usually already correct
        'StockCode': 'StockCode',          # Product identifier
        'Description': 'Description',      # Product description
        'UnitPrice': 'UnitPrice',              # Unit price
        'Quantity': 'Quantity',            # Quantity purchased
        'TransactionDate': 'InvoiceDate'   # Transaction date
    }
    
    # Rename columns to match chatbot expectations
    column_mapping = {}
    for expected, actual in expected_columns.items():
        if actual in final_df.columns:
            column_mapping[actual] = expected
    
    if column_mapping:
        final_df = final_df.rename(columns=column_mapping)
    
    # Clean the data
    # Remove rows with missing essential data
    essential_columns = ['CustomerID', 'StockCode', 'Description', 'UnitPrice']
    final_df = final_df.dropna(subset=essential_columns)
    
    # Remove negative quantities and prices (returns/errors)
    if 'Quantity' in final_df.columns:
        final_df = final_df[final_df['Quantity'] > 0]
    if 'UnitPrice' in final_df.columns:
        final_df = final_df[final_df['UnitPrice'] > 0]
    
    return final_df

def explore_data_for_chatbot(df):
    """
    Explore the data to understand it better for chatbot integration
    """
    if df is None:
        return None
    
    # Basic validation - ensure the data looks correct
    required_columns = ['CustomerID', 'StockCode', 'Description', 'UnitPrice']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
    
    return df

# Example usage
if __name__ == "__main__":
    # CHANGED: Add your Groq API key here
    os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key
    
    # Load and prepare the data
    df = load_ecommerce_data()
    
    if df is not None:
        # Basic validation
        df = explore_data_for_chatbot(df)
        
        # Save the prepared data (optional)
        # df.to_csv('prepared_ecommerce_data.csv', index=False)
        
        print("Data is ready for the chatbot!")
    else:
        print("Could not load data. Please check your CSV files.")
    
    # Initialize your existing CrewAI recommendation system
    recommendation_agents = RecommendationAgents(groq_api_key="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")
    
    # Create the chatbot
    chatbot_api = ChatbotAPI(recommendation_agents, df)
    
    # Example usage:
    session = chatbot_api.start_session("session_123", customer_id=17850)
    print(session['message'])
    
    response = chatbot_api.send_message("session_123", "Show me three similar items to WHITE HANGING HEART T-LIGHT HOLDER")
    print(response['bot_response'])

if __name__ == "__main__":
    # Example usage
    print("E-commerce Conversational AI Chatbot initialized!")
    print("To integrate with your system, use the ChatbotAPI class")
    print("See integrate_with_your_system() function for example usage")