# Fixed src/chatbot/langgraph_chatbot.py

import atexit
import json
import os
import re
import sqlite3
from time import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Tuple
from django.http import JsonResponse
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
import operator
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.processor import DataProcessor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from difflib import SequenceMatcher


_chatbot_instance = None 
FEW_SHOT_EXAMPLES = {
    "greeting": [
        "Hi there!",
        "Hello, my name is Sarah",
        "Good morning! I'm John", 
        "Hey, how are you?",
        "Greetings! I am Mike",
        "Hi, I'm looking to get started",
        "Hello there",
        "Good afternoon"
    ],
    "customer_lookup": [
        "Look up customer 12345",
        "Tell me about customer 67890",
        "Get info on customer 54321", 
        "Show me customer 98765 details",
        "Customer 11111 profile please",
        "Find customer 22222 data",
        "What can you tell me about customer 88888?",
        "Customer 33333 history",
        "Show customer 44444",
        "Get customer 55555 information"
    ],
    "recommendation_request": [
        "Recommend products for customer 12345",
        "What should customer 67890 buy?",
        "Suggest items for customer 54321",
        "Find products similar to RED WIDGET", 
        "Give me recommendations like BLUE GADGET",
        "What products are like GREEN TOOL?",
        "Customer 11111 needs recommendations",
        "Show me 5 recommendations for customer 22222",
        "Products similar to SUPER WIDGET",
        "Recommend something like that"
    ],
    "product_inquiry": [
        "Who bought product WIDGET123?",
        "Who purchased item 45678?",
        "Tell me about product RED WIDGET",
        "Product GADGET456 analysis",
        "Who bought GREEN TOOL?",
        "Analysis of item 78910", 
        "Product 12345 details",
        "Who purchased BLUE WIDGET?",
        "Analysis of product 99999",
        "Product information for 88888"
    ],
    "general_question": [
        "What can you help me with?",
        "How do I use this system?",
        "What are your capabilities?",
        "Help me understand the features",
        "Show me what you can do",
        "Explain how this works", 
        "What functions do you have?",
        "Guide me through the options",
        "What is this about?",
        "Can you help?",
        "Summarize this for me.",
        "explain this",
        "Summarize chat"
    ]
}

# Parameter extraction examples for each intent
PARAMETER_EXAMPLES = {
    "greeting": {
        "examples": [
            ("Hi, I'm John", {"user_name": "John"}),
            ("Hello, my name is Sarah", {"user_name": "Sarah"}),
            ("Hey there!", {}),
            ("Good morning, I am Mike", {"user_name": "Mike"})
        ]
    },
    "customer_lookup": {
        "examples": [
            ("Look up customer 12345", {"customer_id": 12345, "lookup_type": "all"}),
            ("Customer 67890 purchase history", {"customer_id": 67890, "lookup_type": "purchases"}),
            ("Show me customer 54321 behavior", {"customer_id": 54321, "lookup_type": "behavior"}),
            ("Customer 98765 segment info", {"customer_id": 98765, "lookup_type": "segment"})
        ]
    },
    "recommendation_request": [
        ("Recommend for customer 12345", {"customer_id": 12345, "recommendation_count": 5}),
        ("Products like RED WIDGET", {"similar_to_product": "RED WIDGET", "recommendation_count": 5}),
        ("Give me 10 suggestions for customer 67890", {"customer_id": 67890, "recommendation_count": 10}),
        ("Similar to BLUE GADGET", {"similar_to_product": "BLUE GADGET", "recommendation_count": 5})
    ],
    "product_inquiry": [
        ("Who bought WIDGET123?", {"stock_codes": ["WIDGET123"], "inquiry_type": "who bought"}),
        ("Product RED WIDGET analysis", {"product_names": ["RED WIDGET"], "inquiry_type": "all"}),
        ("Tell me about item 45678", {"stock_codes": ["45678"], "inquiry_type": "details"}),
        ("Who purchased GREEN TOOL?", {"product_names": ["GREEN TOOL"], "inquiry_type": "who bought"})
    ],
    "general_question": [
        ("What can you do?", {"help_type": "capabilities"}),
        ("How do I use this?", {"help_type": "usage"}),
        ("Show me examples", {"help_type": "examples"}),
        ("Help me", {"help_type": "general"})
    ]
}

class AgentPromptClassifier:
    """LLM-based agent classifier that can select multiple routes using few-shot prompting."""
    
    def __init__(self, few_shot_examples: Dict[str, List[str]], parameter_examples: Dict[str, List[Tuple]], llm):
        self.few_shot_examples = few_shot_examples
        self.parameter_examples = parameter_examples
        self.intent_classes = list(few_shot_examples.keys())
        self.llm = llm  # Groq LLM instance
    
    def classify_intent(self, prompt: str) -> Dict[str, Any]:
        """Classify prompt intent using LLM agent with few-shot examples."""
        
        # Build few-shot examples for the prompt
        examples_text = ""
        for intent_class, examples in self.few_shot_examples.items():
            examples_text += f"\n{intent_class.upper()} EXAMPLES:\n"
            for example in examples[:3]:  # Use top 3 examples per class
                examples_text += f"- {example}\n"
        
        # Create classification prompt
        classification_prompt = f"""
You are an expert intent classification agent. Analyze the user query and determine which intent(s) it matches.

AVAILABLE INTENTS AND EXAMPLES:
{examples_text}

CLASSIFICATION RULES:
1. A query can match ONE OR MORE intents (multi-intent classification allowed)
2. Return confidence scores for each matched intent (0.0 to 1.0)
3. Only return intents with confidence >= 0.3
4. Consider partial matches and implied intents
5. Use EXACT intent names from the examples above (lowercase with underscores)

USER QUERY: "{prompt}"

Respond in this EXACT JSON format:
{{
    "matched_intents": [
        {{
            "intent": "recommendation_request",
            "confidence": 0.85,
            "reasoning": "brief explanation"
        }},
        {{
            "intent": "customer_lookup",
            "confidence": 0.70,
            "reasoning": "brief explanation"
        }}
    ],
    "primary_intent": "recommendation_request",
    "multi_intent": true
}}

IMPORTANT: Use lowercase intent names with underscores exactly as shown in the examples section headers.
"""
        
        try:
            # Get LLM response
            response = self.llm.invoke([{"role": "user", "content": classification_prompt}])
            result_text = response.content.strip()
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response if wrapped in other text
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
            
            # Process the results
            matched_intents = result_json.get("matched_intents", [])
            primary_intent = result_json.get("primary_intent")
            is_multi_intent = result_json.get("multi_intent", False)
            
            # Create scores dictionary
            all_scores = {intent: 0.0 for intent in self.intent_classes}
            for match in matched_intents:
                intent = match.get("intent")
                confidence = match.get("confidence", 0.0)
                if intent in all_scores:
                    all_scores[intent] = confidence
            
            # Determine best intent if primary not specified or invalid
            if not primary_intent or primary_intent not in self.intent_classes:
                if matched_intents:
                    # Get highest confidence intent
                    best_match = max(matched_intents, key=lambda x: x.get("confidence", 0))
                    primary_intent = best_match["intent"]
                else:
                    primary_intent = "general_question"
            
            # Ensure primary intent is valid
            if primary_intent not in self.intent_classes:
                primary_intent = "general_question"
            
            print(f"🎯 Agent Classification Results:")
            print(f"  Primary Intent: {primary_intent}")
            print(f"  Multi-Intent: {is_multi_intent}")
            print(f"  Matched Intents: {[m['intent'] for m in matched_intents]}")
            print(f"  All Scores: {all_scores}")
            
            return {
                "intent": primary_intent,
                "confidence": all_scores.get(primary_intent, 0.0),
                "all_scores": all_scores,
                "matched_intents": matched_intents,
                "multi_intent": is_multi_intent
            }
            
        except Exception as e:
            print(f"❌ Agent classification error: {e}")
            # Fallback to general_question
            return {
                "intent": "general_question",
                "confidence": 0.5,
                "all_scores": {intent: 0.0 for intent in self.intent_classes},
                "matched_intents": [{"intent": "general_question", "confidence": 0.5, "reasoning": "fallback"}],
                "multi_intent": False
            }
    
    def extract_parameters(self, prompt: str, intent: str) -> Dict[str, Any]:
        """Extract parameters using LLM agent with few-shot examples."""
        if intent not in self.parameter_examples:
            return {}
        
        examples = self.parameter_examples[intent]
        if isinstance(examples, dict) and "examples" in examples:
            examples = examples["examples"]
        
        # Build few-shot parameter examples
        examples_text = ""
        for example_prompt, example_params in examples[:3]:  # Use top 3 examples
            examples_text += f"Query: '{example_prompt}' -> Parameters: {json.dumps(example_params)}\n"
        
        # Create parameter extraction prompt
        extraction_prompt = f"""
You are an expert parameter extraction agent. Extract structured parameters from the user query based on the intent and examples.

INTENT: {intent}

PARAMETER EXTRACTION EXAMPLES:
{examples_text}

USER QUERY: "{prompt}"

Extract parameters following the same pattern as the examples. Return ONLY valid JSON with the extracted parameters.
If no parameters can be extracted, return an empty JSON object {{}}.

JSON Response:
"""
        
        try:
            # Get LLM response
            response = self.llm.invoke([{"role": "user", "content": extraction_prompt}])
            result_text = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                extracted_params = json.loads(json_match.group())
                print(f"📋 Extracted Parameters for {intent}: {extracted_params}")
                return extracted_params
            else:
                return {}
                
        except Exception as e:
            print(f"❌ Parameter extraction error: {e}")
            return {}
    
    def classify_and_extract(self, prompt: str) -> Dict[str, Any]:
        """Main method to classify intent and extract parameters with multi-intent support."""
        # First classify the intent(s)
        classification = self.classify_intent(prompt)
        primary_intent = classification["intent"]
        matched_intents = classification.get("matched_intents", [])
        is_multi_intent = classification.get("multi_intent", False)
        
        # Extract parameters for primary intent
        parameters = self.extract_parameters(prompt, primary_intent)
        
        # If multi-intent, extract parameters for other intents too
        multi_intent_params = {}
        if is_multi_intent and len(matched_intents) > 1:
            for match in matched_intents:
                intent = match["intent"]
                if intent != primary_intent:
                    intent_params = self.extract_parameters(prompt, intent)
                    if intent_params:  # Only add if parameters were found
                        multi_intent_params[intent] = intent_params
        
        return {
            "intent": primary_intent,
            "confidence": classification["confidence"],
            "parameters": parameters,
            "all_intent_scores": classification["all_scores"],
            "matched_intents": matched_intents,
            "multi_intent": is_multi_intent,
            "multi_intent_parameters": multi_intent_params
        }

# Enhanced conversation state with kernel tracking
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: Optional[int]
    context: Dict[str, Any]
    last_recommendations: List[Dict[str, Any]]
    conversation_stage: str
    user_preferences: Dict[str, Any]
    extracted_info: Dict[str, Any]
    last_context: Dict[str, Any]
    # New fields for kernel tracking and RAG
    route_history: List[str]  # Track routing path
    kernel_log: List[Dict[str, Any]]  # Detailed kernel operations
    intermediate_responses: List[str]  # Store intermediate responses for RAG
    final_context: Dict[str, Any]  # Context for final generation agent

class RecommendationChatbot:
    """LangGraph-based chatbot with proper routing, kernel tracking, and RAG-based final generation."""
    
    MAX_HISTORY_MESSAGES = 10 
    # THREAD_ID = "1"  # Fixed thread ID
    # SESSION_ID = "1"  # Fixed session ID

    def __init__(self, groq_api_key: str = None):
        try:
            # Load API key from environment or parameter
            api_key = groq_api_key or os.getenv('GROQ_API_KEY')
            
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables or parameters")
            
            print("🤖 Using Groq API with Llama 3.3 model")
            
            # Initialize Groq LLM for conversation
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1024,
                groq_api_key=api_key
            )

            self.prompt_router = AgentPromptClassifier(FEW_SHOT_EXAMPLES, PARAMETER_EXAMPLES, self.llm)
            print("🎯 Agent-based prompt router initialized")

            # Initialize recommendation tools
            self._init_recommendation_tools()
            
            # Initialize data processor
            print("🔧 Loading data from SQLite database...")
            self.data_processor = DataProcessor()
            self.df_data = None
            self.load_data()
            
            # Initialize database connection
            self.db_path = "db.sqlite3"
            
            # Initialize memory FIRST (before creating graph)
            self.memory = MemorySaver()
            # Add chat history management
            # self.chat_histories = {}        
            # Create the conversation graph (after memory is initialized)
            self.graph = self._create_graph()
            # Add cleanup to prevent memory bloat
            atexit.register(self.cleanup)
            
            print("✅ Chatbot initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            raise

    def cleanup(self):
        """Clean up resources and chat histories."""
        try:
            if hasattr(self, 'memory'):
                del self.memory
                self.memory = MemorySaver()
            
            if hasattr(self, 'chat_histories'):
                self.chat_histories.clear()
            
            if hasattr(self, 'df_data'):
                self.df_data = None
            if hasattr(self, 'vectorizer'):
                self.vectorizer = None
            if hasattr(self, 'tfidf_matrix'):
                self.tfidf_matrix = None
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

    def _log_kernel_operation(self, state: ConversationState, operation: str, details: Dict[str, Any]) -> None:
        """Log kernel operations for tracking."""
        try:
            kernel_entry = {
                "timestamp": pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S'),
                "operation": operation,
                "details": details,
                "conversation_stage": state.get("conversation_stage", "unknown")
            }
            
            if "kernel_log" not in state:
                state["kernel_log"] = []
            
            state["kernel_log"].append(kernel_entry)
            print(f"🔍 Kernel Log: {operation} - {details}")
        except Exception as e:
            print(f"⚠️ Error logging kernel operation: {e}")

    def _init_recommendation_tools(self):
        """Initialize collaborative filtering and content-based recommendation tools."""
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def _prepare_content_vectors(self):
        """Prepare TF-IDF vectors for content-based filtering."""
        try:
            if self.df_data is not None and self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(stop_words="english")
                self.tfidf_matrix = self.vectorizer.fit_transform(self.df_data["Description"])
                print("📊 TF-IDF vectors prepared for content-based filtering")
        except Exception as e:
            print(f"⚠️ Error preparing content vectors: {e}")

    def load_data(self):
        """Load and prepare recommendation data."""
        try:
            df = self.data_processor.load_data_from_sqlite()
            if df is not None:
                self.df_data = self.data_processor.clean_data(df)
                print(f"✅ Loaded {len(self.df_data)} transaction records")
            else:
                print("❌ Failed to load data")
                self.df_data = None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df_data = None

    # ==================== ROUTING DECISION FUNCTION ====================

    def _route_decision(self, state: ConversationState) -> str:
        """Enhanced routing decisions with multi-intent support."""
        try:
            extracted_info = state.get("extracted_info", {})
            primary_intent = extracted_info.get("target", "general_question")
            confidence = extracted_info.get("confidence", 0.0)
            matched_intents = extracted_info.get("matched_intents", [])
            is_multi_intent = extracted_info.get("multi_intent", False)
            
            self._log_kernel_operation(state, "routing_decision", {
                "primary_intent": primary_intent,
                "confidence": confidence,
                "multi_intent": is_multi_intent,
                "matched_intents": [m.get("intent") for m in matched_intents]
            })
            
            # Store multi-intent info for potential sequential processing
            if is_multi_intent and len(matched_intents) > 1:
                state["multi_intent_queue"] = [
                    match["intent"] for match in matched_intents 
                    if match["intent"] != primary_intent and match.get("confidence", 0) >= 0.4
                ]
            
            # Map intents to node names
            intent_node_map = {
                "greeting": "handle_greeting",
                "customer_lookup": "handle_customer_lookup",
                "recommendation_request": "handle_recommendation",
                "product_inquiry": "handle_product_inquiry",
                "general_question": "handle_general_question",
                "parameter_missing": "handle_parameter_missing"
            }
            
            route = intent_node_map.get(primary_intent, "handle_general_question")
            
            self._log_kernel_operation(state, "route_selected", {
                "route": route,
                "is_multi_intent": is_multi_intent
            })
            
            return route
            
        except Exception as e:
            print(f"⚠️ Error in route decision: {e}")
            return "handle_general_question"

    # ==================== HELPER FUNCTIONS ====================

    def _get_comprehensive_customer_data(self, customer_id: int) -> Dict:
        """Get comprehensive customer data for RAG context."""
        try:
            # 1. Get customer basic info
            customer_query = """
            SELECT CustomerID, Country, District, Customer_TotalSpending, Segment
            FROM recommendations_dim_customers 
            WHERE CustomerID = ?
            """
            customer_info = self._execute_db_query(customer_query, (customer_id,))
            
            if not customer_info:
                return {}
            
            customer = customer_info[0]
            
            # 2. Get transaction summary
            transaction_summary_query = """
            SELECT 
                COUNT(DISTINCT InvoiceNo) as total_orders,
                COUNT(*) as total_items,
                SUM(Quantity) as total_quantity,
                AVG(UnitPrice) as avg_item_price,
                MIN(InvoiceDate) as first_purchase,
                MAX(InvoiceDate) as last_purchase,
                COUNT(DISTINCT StockCode_id) as unique_products
            FROM recommendations_fact_transactions 
            WHERE CustomerID_id = ?
            """
            transaction_summary = self._execute_db_query(transaction_summary_query, (customer_id,))
            
            # 3. Get recent purchases (last 10)
            recent_purchases_query = """
            SELECT t.StockCode_id as StockCode, p.Description, t.UnitPrice, 
                   t.Quantity, t.InvoiceDate, p.Description_Categorize
            FROM recommendations_fact_transactions t
            JOIN recommendations_dim_products p ON t.StockCode_id = p.StockCode
            WHERE t.CustomerID_id = ?
            ORDER BY t.InvoiceDate DESC
            LIMIT 10
            """
            recent_purchases = self._execute_db_query(recent_purchases_query, (customer_id,))
            
            # 4. Get top categories
            top_categories_query = """
            SELECT p.Description_Categorize as category, 
                   COUNT(*) as purchase_count,
                   SUM(t.Quantity * t.UnitPrice) as total_spent
            FROM recommendations_fact_transactions t
            JOIN recommendations_dim_products p ON t.StockCode_id = p.StockCode
            WHERE t.CustomerID_id = ?
            GROUP BY p.Description_Categorize
            ORDER BY total_spent DESC
            LIMIT 5
            """
            top_categories = self._execute_db_query(top_categories_query, (customer_id,))
            
            return {
                "customer_info": customer,
                "transaction_summary": transaction_summary[0] if transaction_summary else {},
                "recent_purchases": recent_purchases,
                "top_categories": top_categories
            }
        except Exception as e:
            print(f"⚠️ Error getting customer data: {e}")
            return {}

    def _get_product_info(self, product_query: str) -> List[Dict]:
        """Get product information from database."""
        try:
            # Try to match by stock code first
            stock_code_query = """
            SELECT 
                p.StockCode, 
                p.Description, 
                p.Description_Categorize,
                t.CustomerID_id as customer_id,
                c.Country,
                c.District,
                c.Segment,
                COUNT(t.InvoiceNo) as transactions_per_customer,
                SUM(t.Quantity) as total_quantity_by_customer,
                AVG(t.UnitPrice) as avg_price_for_customer
            FROM recommendations_dim_products p
            INNER JOIN recommendations_fact_transactions t ON p.StockCode = t.StockCode_id
            INNER JOIN recommendations_dim_customers c ON t.CustomerID_id = c.CustomerID
            WHERE p.StockCode LIKE ?
            GROUP BY p.StockCode, p.Description, p.Description_Categorize, t.CustomerID_id, c.Country, c.District, c.Segment
            ORDER BY p.StockCode, t.CustomerID_id
            """
            results = self._execute_db_query(stock_code_query, (f"%{product_query}%",))
            
            if not results:
                # Try to match by description
                desc_query = """
                SELECT p.StockCode, p.Description, p.Description_Categorize,
                    COUNT(t.InvoiceNo) as transaction_count,
                    SUM(t.Quantity) as total_quantity_sold,
                    AVG(t.UnitPrice) as avg_price
                FROM recommendations_dim_products p
                LEFT JOIN recommendations_fact_transactions t ON p.StockCode = t.StockCode_id
                WHERE p.Description LIKE ?
                GROUP BY p.StockCode, p.Description, p.Description_Categorize
                LIMIT 10
                """
                results = self._execute_db_query(desc_query, (f"%{product_query}%",))
            
            return results
        except Exception as e:
            print(f"⚠️ Error getting product info: {e}")
            return []

    def _generate_recommendations_for_customer(self, customer_id: int) -> List[Dict[str, Any]]:
        """Generate recommendations for a customer using collaborative filtering."""
        try:
            if self.df_data is None:
                return []

            # Check if customer exists
            if customer_id not in self.df_data["CustomerID"].unique():
                return []

            # Get customer's recent purchases
            recent_purchases_query = """
            SELECT t.StockCode_id as StockCode, p.Description, t.UnitPrice, t.Quantity
            FROM recommendations_fact_transactions t
            JOIN recommendations_dim_products p ON t.StockCode_id = p.StockCode
            WHERE t.CustomerID_id = ?
            ORDER BY t.InvoiceDate DESC
            LIMIT 3
            """
            recent_purchases = self._execute_db_query(recent_purchases_query, (customer_id,))
            
            if not recent_purchases:
                return []
            
            # Extract stock codes from recent purchases
            stock_codes = [purchase['StockCode'] for purchase in recent_purchases]
            
            # Generate recommendations using collaborative filtering
            collab_results = self._collaborative_filtering(customer_id, stock_codes)
            content_results = self._content_based_filtering(customer_id, stock_codes)
            
            # Combine and rerank recommendations
            recommendations = self._rerank_recommendations(collab_results, content_results, top_n=5)
            
            return recommendations

        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return []

    def _collaborative_filtering(self, target_user_id: int, stock_codes: List[str], top_n: int = 10) -> Dict[str, Any]:
        """Generate recommendations using collaborative filtering."""
        try:
            if self.df_data is None:
                return {"status": "error", "message": "Data not available", "recommendations": []}

            # Check if user exists
            if target_user_id not in self.df_data["CustomerID"].unique():
                return {"status": "error", "message": f"Customer {target_user_id} not found", "recommendations": []}

            # Create user-item matrix
            user_item_matrix = self.df_data.pivot_table(
                index="CustomerID",
                columns="StockCode", 
                values="Quantity",
                aggfunc="mean"
            ).fillna(0)

            # Get user's district for neighborhood filtering
            user_districts = self.df_data[self.df_data["CustomerID"] == target_user_id]["District"]
            if user_districts.empty:
                return {"status": "error", "message": "User district not found", "recommendations": []}
            
            target_district = user_districts.mode().values[0]
            
            # Get users from same district
            district_users = self.df_data[
                (self.df_data["District"] == target_district) & 
                (self.df_data["CustomerID"] != target_user_id)
            ]["CustomerID"].unique()
            
            district_users = [uid for uid in district_users if uid in user_item_matrix.index]

            if not district_users:
                return {"status": "error", "message": "No similar users found", "recommendations": []}

            # Calculate similarities
            filtered_matrix = user_item_matrix.loc[[target_user_id] + district_users]
            similarities = cosine_similarity(filtered_matrix)[0][1:]

            # Get current user items
            current_items = set(user_item_matrix.loc[target_user_id][user_item_matrix.loc[target_user_id] > 0].index)
            updated_items = current_items.union(set(stock_codes))

            # Find recommendations
            recommendations = defaultdict(lambda: {"score": 0, "users": []})
            similar_users = sorted(zip(district_users, similarities), key=lambda x: x[1], reverse=True)[:10]

            for similar_user, similarity in similar_users:
                similar_items = user_item_matrix.loc[similar_user]
                for item in similar_items[similar_items > 0].index:
                    if item not in updated_items:
                        recommendations[item]["score"] += similarity
                        recommendations[item]["users"].append(similar_user)

            # Format results
            result = []
            for item, data in sorted(recommendations.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]:
                try:
                    item_info = self.df_data[self.df_data["StockCode"] == item].iloc[0]
                    result.append({
                        "stock_code": item,
                        "description": item_info["Description"],
                        "unit_price": float(item_info["UnitPrice"]),
                        "score": float(data["score"])
                    })
                except Exception as e:
                    continue

            return {
                "status": "success",
                "target_user_id": target_user_id,
                "district": target_district,
                "recommendations": result
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "recommendations": []}

    def _content_based_filtering(self, target_user_id: int, stock_codes: List[str], recommendations_per_stock: int = 3) -> Dict[str, Any]:
        """Generate recommendations using content-based filtering."""
        try:
            if self.df_data is None:
                return {"status": "error", "message": "Data not available", "recommendations": []}

            # Prepare vectors if not done
            self._prepare_content_vectors()

            if self.tfidf_matrix is None:
                return {"status": "error", "message": "TF-IDF vectors not available", "recommendations": []}

            all_recommendations = []
            seen_descriptions = set()

            for stock_code in stock_codes:
                # Find product index
                indices = self.df_data.index[self.df_data["StockCode"] == stock_code].tolist()
                if not indices:
                    continue

                input_idx = indices[0]
                input_vector = self.tfidf_matrix[input_idx]
                similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

                # Get input product info
                input_description = self.df_data.at[input_idx, "Description"]
                seen_descriptions.add(input_description)

                # Find similar products
                sorted_indices = similarity_scores.argsort()[::-1]
                recommendations = []

                for idx in sorted_indices:
                    if idx == input_idx:
                        continue
                    
                    desc = self.df_data.at[idx, "Description"]
                    if desc not in seen_descriptions:
                        try:
                            recommendations.append({
                                "stock_code": self.df_data.at[idx, "StockCode"],
                                "description": desc,
                                "unit_price": float(self.df_data.at[idx, "UnitPrice"]),
                                "score": float(similarity_scores[idx])
                            })
                            seen_descriptions.add(desc)
                        except Exception as e:
                            continue

                    if len(recommendations) >= recommendations_per_stock:
                        break

                all_recommendations.extend(recommendations)

            return {
                "status": "success",
                "target_user_id": target_user_id,
                "recommendations": all_recommendations
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "recommendations": []}

    def _rerank_recommendations(self, collaborative_results: Dict, content_based_results: Dict, top_n: int = 5) -> List[Dict[str, Any]]:
        """Rerank and combine recommendations."""
        try:
            # Combine all recommendations
            all_recommendations = {}
            
            # Add collaborative filtering results
            if collaborative_results.get("status") == "success":
                for rec in collaborative_results.get("recommendations", []):
                    stock_code = rec["stock_code"]
                    if stock_code not in all_recommendations:
                        all_recommendations[stock_code] = rec.copy()
                    else:
                        all_recommendations[stock_code]["score"] += rec["score"]

            # Add content-based results
            if content_based_results.get("status") == "success":
                for rec in content_based_results.get("recommendations", []):
                    stock_code = rec["stock_code"]
                    if stock_code not in all_recommendations:
                        all_recommendations[stock_code] = rec.copy()
                    else:
                        all_recommendations[stock_code]["score"] += rec["score"]

            # Filter valid recommendations
            valid_recommendations = []
            for stock_code, rec in all_recommendations.items():
                if (rec.get("stock_code") and 
                    rec.get("description") and 
                    rec.get("description").strip() not in ["", "Not Available", "N/A"] and
                    rec.get("unit_price", 0) > 0):
                    valid_recommendations.append(rec)

            # Sort by combined score
            valid_recommendations.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Return top N unique recommendations
            final_recommendations = []
            seen_stock_codes = set()
            
            for rec in valid_recommendations:
                if rec["stock_code"] not in seen_stock_codes and len(final_recommendations) < top_n:
                    final_recommendations.append({
                        "Stock Code": rec["stock_code"],
                        "Description": rec["description"],
                        "Unit Price": rec["unit_price"]
                    })
                    seen_stock_codes.add(rec["stock_code"])

            return final_recommendations

        except Exception as e:
            print(f"❌ Error in reranking: {e}")
            return []

    # ==================== MAIN INTERFACE METHODS ====================

    def chat(self, message: str, chat_id: str = None, user_id: str = "default_user", session_customer_id: int = None) -> str:
        """Main chat interface with Django ChatHistory integration."""
        try:
            # Use chat_id as both thread_id and session_id for isolation
            if not chat_id:
                chat_id = f"chat_{int(time())}"
            
            config = {
                "configurable": {
                    "thread_id": chat_id,
                    "session_id": chat_id
                }
            }
            
            # Load existing chat history from Django model
            existing_messages = self._load_chat_history_from_django(chat_id, session_customer_id)
            
            # Get current state for this specific chat
            current_state = self.graph.get_state(config)
            
            if current_state and current_state.values:
                state = current_state.values
                # Update with loaded messages if state is empty
                if not state.get("messages") and existing_messages:
                    state["messages"] = existing_messages
            else:
                # Create fresh state with loaded history
                state = ConversationState(
                    messages=existing_messages,
                    user_id=session_customer_id,
                    context={"session_customer_id": session_customer_id} if session_customer_id else {},
                    last_recommendations=[],
                    conversation_stage="greeting",
                    user_preferences={},
                    extracted_info={},
                    last_context={},
                    route_history=[],
                    kernel_log=[],
                    intermediate_responses=[],
                    final_context={}
                )
            
            # Update session customer ID in existing state
            if session_customer_id:
                state["user_id"] = session_customer_id
                if "context" not in state:
                    state["context"] = {}
                state["context"]["session_customer_id"] = session_customer_id
            
            # Add user message
            if message.strip():
                state["messages"].append(HumanMessage(content=message))
                
                print(f"Processing message for chat {chat_id} with customer {session_customer_id}: {message[:50]}...")
                
                # Run the graph with the specific config
                result = self.graph.invoke(state, config)
                
                # Save updated conversation to Django
                self._save_chat_history_to_django(chat_id, result["messages"], session_customer_id)
                
                # Get AI response
                ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                if ai_messages:
                    latest_response = ai_messages[-1].content
                    
                    # Print debug info
                    kernel_log = result.get("kernel_log", [])
                    if kernel_log:
                        print(f"\nKERNEL LOG for chat {chat_id}:")
                        for entry in kernel_log[-3:]:
                            print(f"  {entry['timestamp']} - {entry['operation']}: {entry['details']}")
                    
                    return latest_response
            
            return "How can I help you?"
            
        except Exception as e:
            print(f"Error in chat for {chat_id}: {e}")
            return "I encountered an issue. Could you please rephrase your request?"
        
    def _load_chat_history_from_django(self, chat_id: str, session_customer_id: int = None) -> List[BaseMessage]:
        """Load chat history from Django ChatHistory model."""
        try:
            from recommendations.models import ChatHistory
            
            # Try to get existing chat
            try:
                chat_obj = ChatHistory.objects.get(chat_id=chat_id)
                messages_data = chat_obj.get_messages()
                
                # Convert to LangChain messages
                langchain_messages = []
                for msg_data in messages_data:
                    if msg_data.get('type') == 'user':
                        langchain_messages.append(HumanMessage(content=msg_data.get('content', '')))
                    elif msg_data.get('type') == 'assistant':
                        langchain_messages.append(AIMessage(content=msg_data.get('content', '')))
                
                print(f"Loaded {len(langchain_messages)} messages from Django for chat {chat_id}")
                return langchain_messages
                
            except ChatHistory.DoesNotExist:
                print(f"No existing chat history found for chat {chat_id}")
                return []
                
        except Exception as e:
            print(f"Error loading chat history from Django: {e}")
            return []

    def _save_chat_history_to_django(self, chat_id: str, messages: List[BaseMessage], session_customer_id: int = None):
        """Save chat history to Django ChatHistory model."""
        try:
            from recommendations.models import ChatHistory
            
            # Convert LangChain messages to Django format
            messages_data = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    messages_data.append({
                        'type': 'user',
                        'content': msg.content,
                        'timestamp': pd.Timestamp.now(tz='UTC').isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    messages_data.append({
                        'type': 'assistant', 
                        'content': msg.content,
                        'timestamp': pd.Timestamp.now(tz='UTC').isoformat()
                    })
            
            # Generate title from first user message
            title = "New Chat"
            user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            if user_messages:
                first_message = user_messages[0].content
                title = first_message[:50] + "..." if len(first_message) > 50 else first_message
            
            # Update or create chat
            chat_obj, created = ChatHistory.objects.update_or_create(
                chat_id=chat_id,
                defaults={
                    'title': title,
                    'customer_id': session_customer_id,
                }
            )
            
            # Update messages
            chat_obj.set_messages(messages_data)
            chat_obj.save()
            
            print(f"Saved {len(messages_data)} messages to Django for chat {chat_id}")
            
        except Exception as e:
            print(f"Error saving chat history to Django: {e}")

    async def chat_stream(self, message: str, chat_id: str = None, user_id: str = "default_user"):
        """Streaming chat interface with dynamic thread and session IDs per chat."""
        current_timestamp = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
        
        if not chat_id:
            chat_id = f"chat_{int(time())}"
        
        print("\n" + "="*50)
        print(f"🕐 Stream Started at: {current_timestamp}")
        print(f"💬 Chat ID: {chat_id}")
        print(f"👤 User ID: {user_id}")
        print(f"💬 Message: {message}")
        print("="*50 + "\n")

        # Use chat_id for both thread and session isolation
        config = {
            "configurable": {
                "thread_id": chat_id,
                "session_id": chat_id
            }
        }
        
        try:
            # Get current state for this specific chat
            current_state = self.graph.get_state(config)
            if current_state and current_state.values:
                state = current_state.values
            else:
                # Create fresh state for new chat
                state = ConversationState(
                    messages=[],
                    user_id=None,
                    context={},
                    last_recommendations=[],
                    conversation_stage="greeting",
                    user_preferences={},
                    extracted_info={},
                    last_context={},
                    route_history=[],
                    kernel_log=[],
                    intermediate_responses=[],
                    final_context={}
                )
            
            # Add user message
            if message.strip():
                user_message = HumanMessage(content=message)
                state["messages"].append(user_message)
            
            # Stream the response
            buffer = []
            print(f"\n🎯 Streaming messages for chat {chat_id}...")
            
            async for chunk in self.graph.astream(
                state, 
                config, 
                stream_mode="messages"
            ):
                if chunk:
                    token, metadata = chunk
                    buffer.append(token)
                    yield {
                        "token": token,
                        "metadata": metadata,
                        "chat_id": chat_id
                    }
            
            # Log completion
            if buffer:
                complete_response = "".join(str(token) for token in buffer)
                print(f"\n✅ Response completed for chat {chat_id}: {len(buffer)} tokens")
                    
        except Exception as e:
            error_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n❌ Error in chat_stream for {chat_id} at {error_time}: {str(e)}")
            
            yield {
                "error": str(e),
                "timestamp": error_time,
                "chat_id": chat_id,
                "user": user_id
            }


    def clear_chat_history(self, chat_id: str):
        """Clear history for a specific chat."""
        try:
            config = {
                "configurable": {
                    "thread_id": chat_id,
                    "session_id": chat_id
                }
            }
            
            # Create fresh state
            fresh_state = ConversationState(
                messages=[],
                user_id=None,
                context={},
                last_recommendations=[],
                conversation_stage="greeting",
                user_preferences={},
                extracted_info={},
                last_context={},
                route_history=[],
                kernel_log=[],
                intermediate_responses=[],
                final_context={}
            )
            
            # Update the memory with fresh state
            self.graph.update_state(config, fresh_state)
            print(f"✅ Cleared history for chat {chat_id}")
            
        except Exception as e:
            print(f"❌ Error clearing chat history for {chat_id}: {e}")

    def get_chat_context(self, chat_id: str) -> Dict[str, Any]:
        """Get the current context for a specific chat."""
        try:
            config = {
                "configurable": {
                    "thread_id": chat_id,
                    "session_id": chat_id
                }
            }
            
            current_state = self.graph.get_state(config)
            if current_state and current_state.values:
                return {
                    "message_count": len(current_state.values.get("messages", [])),
                    "conversation_stage": current_state.values.get("conversation_stage", "greeting"),
                    "context": current_state.values.get("context", {}),
                    "route_history": current_state.values.get("route_history", []),
                    "has_recommendations": len(current_state.values.get("last_recommendations", [])) > 0
                }
            else:
                return {"message_count": 0, "conversation_stage": "new", "context": {}}
                
        except Exception as e:
            print(f"❌ Error getting chat context for {chat_id}: {e}")
            return {"error": str(e)}

    def _execute_db_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute database query with proper resource cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
                
    def _extract_context_from_messages(self, state: ConversationState) -> Dict[str, Any]:
        """Enhanced context extraction with chat history and session customer ID priority."""
        try:
            self._log_kernel_operation(state, "context_extraction", {"message_count": len(state.get("messages", []))})
            
            # Start with session customer ID if available
            session_customer_id = state.get("context", {}).get("session_customer_id")
            
            context = {
                "mentioned_customer_ids": [],
                "last_customer_id": session_customer_id,
                "session_customer_id": session_customer_id,
                "mentioned_products": [],
                "mentioned_stock_codes": [],
                "last_stock_code": None,
                "query_types": [],
                "last_purchase_info": None,
                "timestamp": time(),
                "chat_history_loaded": True  # Flag to indicate history was considered
            }
            
            # If session customer ID exists, add it to mentioned list
            if session_customer_id:
                context["mentioned_customer_ids"].append(session_customer_id)
            
            # Process ALL messages in chat history (not just recent 5)
            messages = state.get("messages", [])
            unique_messages = []
            seen_contents = set()
            
            # Consider more messages for context extraction from full chat history
            for msg in messages[-10:]:  # Last 10 messages instead of 5
                if isinstance(msg, HumanMessage):
                    content = msg.content.strip()
                    if content and content not in seen_contents:
                        unique_messages.append(msg)
                        seen_contents.add(content)
            
            # Process unique messages for additional customer IDs
            if unique_messages:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for msg in unique_messages:
                        # Extract customer IDs from messages
                        customer_ids = re.findall(r'\b(\d{5,})\b', msg.content)
                        for cid in customer_ids:
                            try:
                                cursor.execute(
                                    "SELECT CustomerID FROM recommendations_dim_customers WHERE CustomerID = ?",
                                    (int(cid),)
                                )
                                if cursor.fetchone():
                                    valid_id = int(cid)
                                    if valid_id not in context["mentioned_customer_ids"]:
                                        context["mentioned_customer_ids"].append(valid_id)
                                        # Update last_customer_id only if no session ID or if explicitly mentioned
                                        if not session_customer_id or valid_id != session_customer_id:
                                            context["last_customer_id"] = valid_id
                            except Exception:
                                continue
            
            return context
            
        except Exception as e:
            print(f"Error in context extraction: {str(e)}")
            return {
                "mentioned_customer_ids": [session_customer_id] if session_customer_id else [],
                "last_customer_id": session_customer_id,
                "session_customer_id": session_customer_id,
                "mentioned_products": [],
                "mentioned_stock_codes": [],
                "last_stock_code": None,
                "query_types": [],
                "last_purchase_info": None,
                "timestamp": time(),
                "chat_history_loaded": False
            }


    def _create_graph(self) -> StateGraph:
        """Create the LangGraph conversation flow with enhanced prompt target routing."""
        try:
            # Define the graph
            workflow = StateGraph(ConversationState)

            # Add nodes
            workflow.add_node("route_query", self._route_query_node)
            workflow.add_node("handle_greeting", self._handle_greeting_node)
            workflow.add_node("handle_customer_lookup", self._handle_customer_lookup_node)
            workflow.add_node("handle_recommendation", self._handle_recommendation_node)
            workflow.add_node("handle_product_inquiry", self._handle_product_inquiry_node)
            workflow.add_node("handle_general_question", self._handle_general_question_node)
            workflow.add_node("handle_parameter_missing", self._handle_parameter_missing_node)  # New node
            workflow.add_node("final_generation", self._final_generation_node)
            
            # Define the conversation flow
            workflow.set_entry_point("route_query")
            
            # Add conditional edges from route_query to specific handlers
            workflow.add_conditional_edges(
                "route_query",
                self._route_decision,
                {
                    "handle_greeting": "handle_greeting",
                    "handle_customer_lookup": "handle_customer_lookup",
                    "handle_recommendation": "handle_recommendation",
                    "handle_product_inquiry": "handle_product_inquiry",
                    "handle_general_question": "handle_general_question",
                    "handle_parameter_missing": "handle_parameter_missing",  # New route
                    "end": END
                }
            )
            
            # All handlers flow to final generation agent (RAG)
            workflow.add_edge("handle_greeting", "final_generation")
            workflow.add_edge("handle_customer_lookup", "final_generation")
            workflow.add_edge("handle_recommendation", "final_generation")
            workflow.add_edge("handle_product_inquiry", "final_generation")
            workflow.add_edge("handle_general_question", "final_generation")
            workflow.add_edge("handle_parameter_missing", "final_generation")  # New edge
            
            # Final generation flows to END
            workflow.add_edge("final_generation", END)
            
            return workflow.compile(checkpointer=self.memory)
            
        except Exception as e:
            print(f"❌ Error creating graph: {e}")
            raise

    # ==================== NODE IMPLEMENTATIONS ====================

    def _route_query_node(self, state: ConversationState) -> ConversationState:
        """Enhanced route query node using agent-based classification."""
        try:
            self._log_kernel_operation(state, "agent_classification_start", {})
            
            # Initialize route history if not exists
            if "route_history" not in state:
                state["route_history"] = []
            
            # Get the last user message
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if not user_messages:
                self._log_kernel_operation(state, "routing_failed", {"reason": "no_user_messages"})
                state["conversation_stage"] = "general_question"
                return state

            last_message = user_messages[-1].content
            
            # DEBUG: Print the message being classified
            print(f"\n🤖 AGENT CLASSIFYING MESSAGE: '{last_message}'")
            
            # Use agent-based classifier
            routing_result = self.prompt_router.classify_and_extract(last_message)
            
            primary_intent = routing_result["intent"]
            confidence = routing_result["confidence"]
            extracted_params = routing_result["parameters"]
            matched_intents = routing_result.get("matched_intents", [])
            is_multi_intent = routing_result.get("multi_intent", False)
            multi_intent_params = routing_result.get("multi_intent_parameters", {})
            
            # DEBUG: Print classification results
            print(f"🎯 AGENT CLASSIFICATION RESULT:")
            print(f"  Primary Intent: {primary_intent}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Parameters: {extracted_params}")
            print(f"  Multi-Intent: {is_multi_intent}")
            if is_multi_intent:
                print(f"  All Matched Intents: {[m.get('intent') for m in matched_intents]}")
                print(f"  Multi-Intent Params: {multi_intent_params}")
            
            # Map intent names to target names for compatibility
            intent_target_map = {
                "greeting": "greeting",
                "customer_lookup": "customer_lookup", 
                "recommendation_request": "recommendation_request",
                "product_inquiry": "product_inquiry",
                "general_question": "general_question"
            }
            
            mapped_target = intent_target_map.get(primary_intent, "general_question")
            
            # Store routing information with multi-intent support
            state["extracted_info"] = {
                "target": mapped_target,
                "confidence": confidence,
                "extracted_parameters": extracted_params,
                "original_query": last_message,
                "timestamp": pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S'),
                # NEW: Multi-intent support
                "matched_intents": matched_intents,
                "multi_intent": is_multi_intent,
                "multi_intent_parameters": multi_intent_params
            }
            
            # Update conversation stage and context
            state["conversation_stage"] = mapped_target
            state["context"].update(extracted_params)
            
            # Add multi-intent parameters to context
            if multi_intent_params:
                state["context"]["multi_intent_params"] = multi_intent_params
            
            state["route_history"].append(f"{mapped_target}")
            if is_multi_intent:
                state["route_history"].append(f"multi_intent({len(matched_intents)})")
            
            self._log_kernel_operation(state, "agent_classification_complete", {
                "primary_target": mapped_target,
                "confidence": confidence,
                "extracted_params": list(extracted_params.keys()),
                "route_count": len(state["route_history"]),
                "is_multi_intent": is_multi_intent,
                "matched_intents_count": len(matched_intents)
            })
            
            return state
            
        except Exception as e:
            print(f"❌ AGENT CLASSIFICATION ERROR: {e}")
            self._log_kernel_operation(state, "agent_classification_error", {"error": str(e)})
            state["conversation_stage"] = "general_question"
            return state
        
    def _handle_parameter_missing_node(self, state: ConversationState) -> ConversationState:
        """Handle cases where required parameters are missing."""
        try:
            self._log_kernel_operation(state, "parameter_missing_handling", {})
            
            parameter_issues = state.get("parameter_issues", {})
            missing_params = parameter_issues.get("missing_parameters", [])
            target_config = parameter_issues.get("target_config", {})
            
            # Create helpful prompt for missing parameters
            param_descriptions = {}
            for param in target_config.get("parameters", []):
                if param["name"] in missing_params:
                    param_descriptions[param["name"]] = param["description"]
            
            # Store intermediate response for RAG
            intermediate_response = f"""
    PARAMETER_MISSING_CONTEXT: Required parameters are missing for the requested action.
    TARGET_NAME: {target_config.get('name', 'unknown')}
    TARGET_DESCRIPTION: {target_config.get('description', '')}
    MISSING_PARAMETERS: {missing_params}
    PARAMETER_DESCRIPTIONS: {param_descriptions}
    RESPONSE_NEEDED: Ask user to provide the missing required information in a helpful way.
            """
            
            if "intermediate_responses" not in state:
                state["intermediate_responses"] = []
            state["intermediate_responses"].append(intermediate_response)
            
            # Set context for final generation
            state["final_context"] = {
                "interaction_type": "parameter_missing",
                "missing_parameters": missing_params,
                "parameter_descriptions": param_descriptions,
                "target_name": target_config.get("name"),
                "target_description": target_config.get("description")
            }
            
            self._log_kernel_operation(state, "parameter_missing_processed", {
                "missing_count": len(missing_params)
            })
            
        except Exception as e:
            self._log_kernel_operation(state, "parameter_missing_error", {"error": str(e)})
        
        return state

    def _handle_greeting_node(self, state: ConversationState) -> ConversationState:
        """Handle greeting messages with session customer awareness."""
        try:
            session_customer_id = state.get("context", {}).get("session_customer_id")
            self._log_kernel_operation(state, "greeting_handling", {
                "user_id": state.get("user_id"),
                "session_customer_id": session_customer_id
            })
            
            # Extract user name if mentioned
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            last_message = user_messages[-1].content if user_messages else ""
            
            user_name = ""
            name_match = re.search(r"(?:i'm|my name is|i am)\s+(\w+)", last_message, re.IGNORECASE)
            if name_match:
                user_name = name_match.group(1)
                state["context"]["user_name"] = user_name
            
            # Get customer data if session customer ID is available
            customer_data = {}
            if session_customer_id:
                customer_data = self._get_comprehensive_customer_data(session_customer_id)
            
            # Store intermediate response for RAG
            intermediate_response = f"""
GREETING_CONTEXT: User said hello{f' and introduced themselves as {user_name}' if user_name else ''}. 
This is a greeting interaction that should result in a welcoming response.
SESSION_CUSTOMER_ID: {session_customer_id if session_customer_id else 'Not logged in'}
CUSTOMER_DATA: {json.dumps(customer_data, indent=2) if customer_data else 'No customer data available'}
CAPABILITIES: The assistant can help with product recommendations, customer analysis, purchase history, and product information.
TONE: Should be friendly, welcoming, and personalized if customer data is available.
            """
            
            if "intermediate_responses" not in state:
                state["intermediate_responses"] = []
            state["intermediate_responses"].append(intermediate_response)
            
            # Set context for final generation
            state["final_context"] = {
                "interaction_type": "greeting",
                "user_name": user_name,
                "session_customer_id": session_customer_id,
                "customer_data": customer_data,
                "should_introduce_capabilities": True,
                "tone": "friendly_welcoming_personalized" if session_customer_id else "friendly_welcoming"
            }
            
            self._log_kernel_operation(state, "greeting_processed", {
                "has_name": bool(user_name),
                "has_session_customer": bool(session_customer_id)
            })
            
        except Exception as e:
            self._log_kernel_operation(state, "greeting_error", {"error": str(e)})
        
        return state

    def _handle_customer_lookup_node(self, state: ConversationState) -> ConversationState:
        """Enhanced customer lookup with extracted parameters."""
        try:
            self._log_kernel_operation(state, "customer_lookup_start", {})
            
            extracted_info = state.get("extracted_info", {})
            extracted_params = extracted_info.get("extracted_parameters", {})
            
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            original_query = user_messages[-1].content if user_messages else ""
            
            # Use extracted customer_id if available
            user_id = extracted_params.get("customer_id")
            lookup_type = extracted_params.get("lookup_type", "all")
            
            # Fallback to context extraction if no parameters extracted
            if not user_id:
                context = self._extract_context_from_messages(state)
                user_id = context.get("last_customer_id")
            
            if user_id:
                # Get customer data
                customer_data = self._get_comprehensive_customer_data(user_id)
                
                if customer_data.get("customer_info"):
                    # Store intermediate response for RAG with lookup type
                    intermediate_response = f"""
    CUSTOMER_LOOKUP_CONTEXT: User requested {lookup_type} information about customer {user_id}.
    CUSTOMER_DATA: {json.dumps(customer_data, indent=2)}
    LOOKUP_TYPE: {lookup_type}
    EXTRACTED_PARAMETERS: {extracted_params}
    QUERY_TYPE: {original_query}
    RESPONSE_NEEDED: Provide relevant customer information based on lookup type and query.
                    """
                    
                    if "intermediate_responses" not in state:
                        state["intermediate_responses"] = []
                    state["intermediate_responses"].append(intermediate_response)
                    
                    # Set context for final generation
                    state["final_context"] = {
                        "interaction_type": "customer_lookup",
                        "customer_id": user_id,
                        "customer_data": customer_data,
                        "lookup_type": lookup_type,
                        "original_query": original_query,
                        "extracted_parameters": extracted_params,
                        "has_data": True
                    }
                    
                    self._log_kernel_operation(state, "customer_data_found", {
                        "customer_id": user_id,
                        "lookup_type": lookup_type
                    })
                else:
                    # Customer not found - same as before
                    intermediate_response = f"""
    CUSTOMER_LOOKUP_CONTEXT: Customer {user_id} not found in database.
    RESPONSE_NEEDED: Inform user that customer ID was not found and suggest checking the ID.
                    """
                    
                    if "intermediate_responses" not in state:
                        state["intermediate_responses"] = []
                    state["intermediate_responses"].append(intermediate_response)
                    
                    state["final_context"] = {
                        "interaction_type": "customer_lookup",
                        "customer_id": user_id,
                        "has_data": False,
                        "error": "customer_not_found"
                    }
                    
                    self._log_kernel_operation(state, "customer_not_found", {"customer_id": user_id})
            else:
                # No customer ID provided - same as before
                intermediate_response = """
    CUSTOMER_LOOKUP_CONTEXT: User requested customer information but no valid customer ID was provided.
    RESPONSE_NEEDED: Ask user to provide a customer ID for lookup.
                """
                
                if "intermediate_responses" not in state:
                    state["intermediate_responses"] = []
                state["intermediate_responses"].append(intermediate_response)
                
                state["final_context"] = {
                    "interaction_type": "customer_lookup",
                    "has_data": False,
                    "error": "no_customer_id"
                }
                
                self._log_kernel_operation(state, "no_customer_id", {})
            
        except Exception as e:
            self._log_kernel_operation(state, "customer_lookup_error", {"error": str(e)})
        
        return state

    def _handle_recommendation_node(self, state: ConversationState) -> ConversationState:
        """Enhanced recommendation handling with session customer ID priority."""
        try:
            session_customer_id = state.get("context", {}).get("session_customer_id")
            self._log_kernel_operation(state, "recommendation_start", {"session_customer_id": session_customer_id})
            
            extracted_info = state.get("extracted_info", {})
            extracted_params = extracted_info.get("extracted_parameters", {})
            
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            original_query = user_messages[-1].content if user_messages else ""
            
            # Priority order: session customer ID > extracted customer ID > context customer ID
            user_id = session_customer_id or extracted_params.get("customer_id")
            similar_to_product = extracted_params.get("similar_to_product")
            recommendation_count = extracted_params.get("recommendation_count", 5)
            
            # Fallback to context extraction only if no session or extracted customer ID
            if not user_id and not similar_to_product:
                context = self._extract_context_from_messages(state)
                user_id = context.get("last_customer_id")
            
            recommendations = []
            recommendation_type = "general"
            
            if similar_to_product:
                # Handle product similarity recommendations
                recommendations = self._get_similar_products(similar_to_product, recommendation_count)
                recommendation_type = "product_similarity"
                
                self._log_kernel_operation(state, "similar_product_recommendations", {
                    "product": similar_to_product,
                    "count": len(recommendations)
                })
                
            elif user_id and self.df_data is not None:
                # Handle customer-based recommendations
                if user_id in self.df_data["CustomerID"].unique():
                    recommendations = self._generate_recommendations_for_customer(user_id)
                    recommendation_type = "customer_based"
                    
                    self._log_kernel_operation(state, "customer_recommendations_generated", {
                        "customer_id": user_id,
                        "is_session_customer": user_id == session_customer_id,
                        "count": len(recommendations)
                    })
                else:
                    self._log_kernel_operation(state, "recommendation_customer_not_found", {"customer_id": user_id})
            
            # Store intermediate response for RAG
            if recommendations:
                intermediate_response = f"""
    RECOMMENDATION_CONTEXT: User requested {recommendation_type} recommendations.
    ORIGINAL_QUERY: {original_query}
    SESSION_CUSTOMER_ID: {session_customer_id}
    USED_CUSTOMER_ID: {user_id}
    IS_SESSION_CUSTOMER: {user_id == session_customer_id if session_customer_id else False}
    EXTRACTED_PARAMETERS: {extracted_params}
    RECOMMENDATIONS_DATA: {json.dumps(recommendations, indent=2)}
    RECOMMENDATION_TYPE: {recommendation_type}
    RESPONSE_NEEDED: Present the recommendations in a user-friendly format with explanations.
                """
            else:
                intermediate_response = f"""
    RECOMMENDATION_CONTEXT: User requested recommendations but none could be generated.
    ORIGINAL_QUERY: {original_query}
    SESSION_CUSTOMER_ID: {session_customer_id}
    EXTRACTED_PARAMETERS: {extracted_params}
    ERROR: No recommendations available
    RESPONSE_NEEDED: Explain why no recommendations could be generated. If logged in, provide personalized guidance.
                """
            
            if "intermediate_responses" not in state:
                state["intermediate_responses"] = []
            state["intermediate_responses"].append(intermediate_response)
            
            # Set context for final generation
            state["final_context"] = {
                "interaction_type": "recommendation",
                "session_customer_id": session_customer_id,
                "customer_id": user_id,
                "is_session_customer": user_id == session_customer_id if session_customer_id else False,
                "similar_to_product": similar_to_product,
                "recommendations": recommendations,
                "recommendation_type": recommendation_type,
                "original_query": original_query,
                "extracted_parameters": extracted_params,
                "has_recommendations": len(recommendations) > 0
            }
            
            state["last_recommendations"] = recommendations
            
        except Exception as e:
            self._log_kernel_operation(state, "recommendation_error", {"error": str(e)})
        
        return state
    
    def _get_similar_products(self, product_name: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get similar products using content-based filtering."""
        try:
            if self.df_data is None:
                return []
            
            # Find the product in the dataframe
            product_matches = self.df_data[
                self.df_data["Description"].str.contains(product_name, case=False, na=False)
            ]
            
            if product_matches.empty:
                return []
            
            # Use the first match as reference
            reference_product = product_matches.iloc[0]
            reference_desc = reference_product["Description"]
            
            # Prepare vectors if not done
            self._prepare_content_vectors()
            
            if self.tfidf_matrix is None:
                return []
            
            # Find the index of the reference product
            ref_index = reference_product.name
            if ref_index >= len(self.tfidf_matrix.toarray()):
                return []
            
            # Calculate similarities
            ref_vector = self.tfidf_matrix[ref_index]
            similarities = cosine_similarity(ref_vector, self.tfidf_matrix).flatten()
            
            # Get top similar products (excluding the reference)
            similar_indices = similarities.argsort()[::-1]
            
            recommendations = []
            seen_descriptions = {reference_desc}
            
            for idx in similar_indices:
                if len(recommendations) >= count:
                    break
                
                if idx == ref_index:
                    continue
                
                try:
                    product = self.df_data.iloc[idx]
                    desc = product["Description"]
                    
                    if desc not in seen_descriptions:
                        recommendations.append({
                            "Stock Code": product["StockCode"],
                            "Description": desc,
                            "Unit Price": float(product["UnitPrice"])
                        })
                        seen_descriptions.add(desc)
                except Exception:
                    continue
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting similar products: {e}")
            return []

    def _handle_product_inquiry_node(self, state: ConversationState) -> ConversationState:
        """Enhanced product inquiry handling with stock code analysis and customer lookup."""
        try:
            self._log_kernel_operation(state, "product_inquiry_start", {})
            
            extracted_info = state.get("extracted_info", {})
            extracted_params = extracted_info.get("extracted_parameters", {})
            
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            original_query = user_messages[-1].content if user_messages else ""
            
            # Extract product references from parameters or fallback to manual extraction
            stock_codes = extracted_params.get("stock_codes", [])
            product_names = extracted_params.get("product_names", [])
            inquiry_type = extracted_params.get("inquiry_type", "all")
            
            # Manual extraction if parameters missed something
            if not stock_codes:
                manual_codes = re.findall(r'\b(\d{4,8}|[A-Z0-9]{4,8})\b', original_query)
                stock_codes.extend(manual_codes)
            
            if not product_names:
                manual_names = re.findall(r'(?:"([^"]+)"|([A-Z][A-Z\s&]+[A-Z]))', original_query)
                if manual_names:
                    flattened_names = [name for group in manual_names for name in group if name]
                    product_names.extend(flattened_names)
            
            product_refs = stock_codes + product_names
            
            if product_refs:
                # Get product information
                products_data = []
                customers_data = []
                
                for ref in product_refs[:3]:  # Limit to 3 products
                    # Get product details
                    products = self._get_product_info(ref)
                    products_data.extend(products)
                
                # Store intermediate response for RAG
                intermediate_response = f"""
    PRODUCT_INQUIRY_CONTEXT: User asked about product details and analysis.
    ORIGINAL_QUERY: {original_query}
    PRODUCT_REFERENCES: {product_refs}
    INQUIRY_TYPE: {inquiry_type}
    PRODUCTS_DATA: {json.dumps(products_data, indent=2)}
    CUSTOMERS_DATA: {json.dumps(customers_data, indent=2) if customers_data else "No customer data requested"}
    RESPONSE_NEEDED: Present product information and customer analysis in a clear, detailed format.
                """
                
                if "intermediate_responses" not in state:
                    state["intermediate_responses"] = []
                state["intermediate_responses"].append(intermediate_response)
                
                # Set context for final generation
                state["final_context"] = {
                    "interaction_type": "product_inquiry",
                    "products_data": products_data,
                    "customers_data": customers_data,
                    "product_references": product_refs,
                    "inquiry_type": inquiry_type,
                    "original_query": original_query,
                    "has_products": len(products_data) > 0,
                    "has_customers": len(customers_data) > 0
                }
                
                self._log_kernel_operation(state, "products_and_customers_found", {
                    "products_count": len(products_data),
                    "customers_count": len(customers_data)
                })
            else:
                # No products found
                intermediate_response = f"""
    PRODUCT_INQUIRY_CONTEXT: User asked about products but no specific products could be identified.
    ORIGINAL_QUERY: {original_query}
    RESPONSE_NEEDED: Ask user to be more specific about which products they want information about.
                """
                
                if "intermediate_responses" not in state:
                    state["intermediate_responses"] = []
                state["intermediate_responses"].append(intermediate_response)
                
                state["final_context"] = {
                    "interaction_type": "product_inquiry",
                    "has_products": False,
                    "original_query": original_query,
                    "error": "no_products_identified"
                }
                
                self._log_kernel_operation(state, "no_products_identified", {})
            
        except Exception as e:
            self._log_kernel_operation(state, "product_inquiry_error", {"error": str(e)})
        
        return state

    def _handle_general_question_node(self, state: ConversationState) -> ConversationState:
        """Handle general questions and help requests."""
        try:
            self._log_kernel_operation(state, "general_question_start", {})
            
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            original_query = user_messages[-1].content if user_messages else ""
            
            # Store intermediate response for RAG
            intermediate_response = f"""
GENERAL_QUESTION_CONTEXT: User asked a general question or requested help.
ORIGINAL_QUERY: {original_query}
**CRITICAL INSTRUCTIONS:**
- Your knowledge is STRICTLY LIMITED to the following domains: product recommendations, customer analysis, purchase history, and general product information.
- If the user's question is completely unrelated to these domains (e.g., about astronomy, history, sports scores, other companies, or general knowledge), you MUST politely decline to answer.
- In such cases, do not attempt to answer. Instead, clearly state that you are a specialized assistant for this company and its products.
RESPONSE_NEEDED: Provide helpful information about the assistant's capabilities or answer the general question.
CAPABILITIES: Product recommendations, customer analysis, purchase history, product information.
            """
            
            if "intermediate_responses" not in state:
                state["intermediate_responses"] = []
            state["intermediate_responses"].append(intermediate_response)
            
            # Set context for final generation
            state["final_context"] = {
                "interaction_type": "general_question",
                "original_query": original_query,
                "should_explain_capabilities": True
            }
            
            self._log_kernel_operation(state, "general_question_processed", {})
            
        except Exception as e:
            self._log_kernel_operation(state, "general_question_error", {"error": str(e)})
        
        return state

    def _final_generation_node(self, state: ConversationState) -> ConversationState:
        """RAG-based final generation agent that creates the final response."""
        try:
            self._log_kernel_operation(state, "final_generation_start", {})
            
            # Get all context and intermediate responses
            final_context = state.get("final_context", {})
            intermediate_responses = state.get("intermediate_responses", [])
            route_history = state.get("route_history", [])
            kernel_log = state.get("kernel_log", [])
            
            # Get current message and conversation history
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            
            current_query = user_messages[-1].content if user_messages else ""
            
            # Build comprehensive context for RAG
            rag_context = f"""
CURRENT_USER_QUERY: {current_query}

ROUTE_HISTORY: {' -> '.join(route_history)}

INTERACTION_TYPE: {final_context.get('interaction_type', 'unknown')}

INTERMEDIATE_CONTEXT:
{chr(10).join(intermediate_responses)}

CONVERSATION_HISTORY:
"""
            
            # Add recent conversation history (last 4 exchanges)
            recent_messages = state["messages"][-8:]  # Last 4 user-ai exchanges
            for i, msg in enumerate(recent_messages):
                if isinstance(msg, HumanMessage):
                    rag_context += f"USER: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    rag_context += f"ASSISTANT: {msg.content}\n"
            
            # Create the RAG prompt
            rag_prompt = f"""
You are an expert e-commerce assistant with access to comprehensive context about the current interaction.

CONTEXT INFORMATION:
{rag_context}

INSTRUCTIONS:
1. Analyze the current user query and all provided context
2. Generate a helpful, accurate, and engaging response
3. Use the interaction type to determine the appropriate response style
4. If data was found, present it clearly and informatively
5. If errors occurred, handle them gracefully
6. Maintain a conversational and professional tone
7. Include specific details from the context when relevant
8. Format the response with appropriate emojis and structure
9. Ensure the response is concise and to the point
10. Don't answer questions outside of the provided context

RESPONSE GUIDELINES:
- For greetings: Be welcoming and introduce capabilities
- For customer lookups: Present customer data clearly with insights
- For recommendations: Format recommendations attractively with explanations
- For product inquiries: Provide detailed product information
- For general questions: Be helpful and informative
- For errors: Gracefully explain the issue and offer alternatives

Generate the final response for the user:
            """
            
            # Generate the final response using the LLM
            response = self.llm.invoke([HumanMessage(content=rag_prompt)])
            final_response = response.content
            
            # Add the final response to messages
            state["messages"].append(AIMessage(content=final_response))
            
            self._log_kernel_operation(state, "final_generation_complete", {
                "response_length": len(final_response),
                "interaction_type": final_context.get("interaction_type")
            })

        except Exception as e:
            self._log_kernel_operation(state, "final_generation_error", {
                "error": str(e),
                "interaction_type": final_context.get("interaction_type")
            })
            # Fallback response
            fallback_response = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question."
            state["messages"].append(AIMessage(content=fallback_response))
        
        return state

# ==================== DJANGO VIEW ====================

# async def create_chatbot_view(request):
#     """Django view for the chatbot interface with streaming support."""
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             message = data.get('message', '')
#             user_id = data.get('user_id', 'default')
            
#             global _chatbot_instance
#             if _chatbot_instance is None:
#                 _chatbot_instance = RecommendationChatbot()
            
#             # For streaming, return a streaming response
#             from django.http import StreamingHttpResponse
            
#             async def generate():
#                 try:
#                     async for chunk in _chatbot_instance.chat_stream(message, user_id):
#                         yield f"data: {json.dumps({'chunk': chunk})}\n\n"
#                 except Exception as e:
#                     yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
#             return StreamingHttpResponse(
#                 generate(),
#                 content_type='text/event-stream'
#             )
#         except Exception as e:
#             return JsonResponse({'error': f'Request processing error: {str(e)}'}, status=500)

#     return JsonResponse({'error': 'Only POST requests allowed'}, status=405)