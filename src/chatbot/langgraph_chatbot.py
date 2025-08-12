# Updated src/chatbot/langgraph_chatbot.py

import json
# import logging
import os
import re
import sqlite3
from typing import Dict, List, Any, Optional, TypedDict, Annotated
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
# Import your existing CrewAI agents
# from src.agents.crew_agents import RecommendationAgents
from src.data.processor import DataProcessor
_chatbot_instance = None 
# Define the conversation state
# Line 21-29: Update ConversationState
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: Optional[int]
    context: Dict[str, Any]
    last_recommendations: List[Dict[str, Any]]
    conversation_stage: str
    user_preferences: Dict[str, Any]
    extracted_info: Dict[str, Any]
    # conversation_history: List[Dict[str, Any]]  # NEW: Add conversation history

class RecommendationChatbot:

    # Get conversation history automatically
    def _extract_context_from_messages(self, state: ConversationState) -> Dict[str, Any]:
        """Extract context from built-in message history."""
        context = {
            "mentioned_customer_ids": [],
            "last_customer_id": None,
            "mentioned_products": [],
            "mentioned_stock_codes": [],
            "last_stock_code": None,
            "query_types": []
        }
        
        # Get ALL messages from persistent state
        # Get RECENT messages for context (not all 38k messages!)
        all_messages = state["messages"]
        recent_messages = all_messages[-20:]  # Last 20 messages for context
        print(f"🔍 Context extraction: Analyzing {len(recent_messages)} recent messages from {len(all_messages)} total")
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                # Extract customer IDs (5+ digits)
                customer_id_matches = re.findall(r'\b(\d{5,})\b', msg.content)
                for match in customer_id_matches:
                    customer_id = int(match)
                    if customer_id not in context["mentioned_customer_ids"]:
                        context["mentioned_customer_ids"].append(customer_id)
                        context["last_customer_id"] = customer_id
                
                # Extract stock codes (better pattern - actual stock code format)
                stock_code_matches = re.findall(r'\b([0-9]{4,6}[A-Z]?|[A-Z]{1,3}[0-9]{3,6})\b', msg.content.upper())
                for stock_code in stock_code_matches:
                    if stock_code not in context["mentioned_stock_codes"]:
                        context["mentioned_stock_codes"].append(stock_code)
                        context["last_stock_code"] = stock_code
                
                # Extract quoted product names only
                product_matches = re.findall(r'"([^"]{5,})"', msg.content)
                for product in product_matches:
                    clean_product = product.strip()
                    if clean_product not in context["mentioned_products"]:
                        context["mentioned_products"].append(clean_product)
                            
            elif isinstance(msg, AIMessage):
                # Skip greeting messages
                if any(phrase in msg.content.lower() for phrase in [
                    "hello", "i'm your ai", "just tell me", "for example"
                ]):
                    continue
                    
                # Extract customer IDs from actual AI responses
                customer_id_matches = re.findall(r'\bcustomer\s+(\d{5,})\b', msg.content, re.IGNORECASE)
                for match in customer_id_matches:
                    customer_id = int(match)
                    if customer_id not in context["mentioned_customer_ids"]:
                        context["mentioned_customer_ids"].append(customer_id)
                        context["last_customer_id"] = customer_id
        
        return context
    """LangGraph-based chatbot that integrates with CrewAI recommendation agents."""
    
    def __init__(self, groq_api_key: str = None):
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
        
        # Initialize CrewAI agents with the same API key
        # self.crew_agents = RecommendationAgents(groq_api_key=api_key)
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
        
        # Create the conversation graph (after memory is initialized)
        self.graph = self._create_graph()

    def _init_recommendation_tools(self):
        """Initialize collaborative filtering and content-based recommendation tools."""
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def _prepare_content_vectors(self):
        """Prepare TF-IDF vectors for content-based filtering."""
        if self.df_data is not None and self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df_data["Description"])
            print("📊 TF-IDF vectors prepared for content-based filtering")

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
            target_district = self.df_data[self.df_data["CustomerID"] == target_user_id]["District"].mode().values[0]
            
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
                        # "source": "Collaborative Filtering",
                        # "popularity": len(data["users"]),
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
                                # "source": "Content-Based",
                                # "popularity": 0,
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
        """LangGraph agent to rerank and combine recommendations."""
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
                        # Mark as "Both" if from multiple sources
                        all_recommendations[stock_code]["source"] = "Both"
                        all_recommendations[stock_code]["score"] += rec["score"]

            # Add content-based results
            if content_based_results.get("status") == "success":
                for rec in content_based_results.get("recommendations", []):
                    stock_code = rec["stock_code"]
                    if stock_code not in all_recommendations:
                        all_recommendations[stock_code] = rec.copy()
                    else:
                        # Mark as "Both" and combine scores
                        all_recommendations[stock_code]["source"] = "Both"
                        all_recommendations[stock_code]["score"] += rec["score"]

            # Filter valid recommendations
            valid_recommendations = []
            for stock_code, rec in all_recommendations.items():
                # Validate recommendation data
                if (rec.get("stock_code") and 
                    rec.get("description") and 
                    rec.get("description").strip() not in ["", "Not Available", "N/A"] and
                    rec.get("unit_price", 0) > 0):
                    valid_recommendations.append(rec)

            # Sort by combined score (collaborative score + content score)
            valid_recommendations.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Return top N unique recommendations
            final_recommendations = []
            seen_stock_codes = set()
            
            for rec in valid_recommendations:
                if rec["stock_code"] not in seen_stock_codes and len(final_recommendations) < top_n:
                    final_recommendations.append({
                        "Stock Code": rec["stock_code"],
                        "Description": rec["description"],
                        "Unit Price": rec["unit_price"],
                        # "Source": rec["source"],
                        # "Popularity": rec.get("popularity", 0)
                    })
                    seen_stock_codes.add(rec["stock_code"])

            return final_recommendations

        except Exception as e:
            print(f"❌ Error in reranking: {e}")
            return []

    def _generate_recommendations_node(self, state: ConversationState) -> ConversationState:
        """Generate recommendations using LangGraph pipeline instead of CrewAI."""
        try:
            user_id = state.get("user_id")
            extracted = state.get("extracted_info", {})
            stock_codes = extracted.get("stock_codes", [])
            specific_products = extracted.get("specific_products", [])
            
            # Get original query for context
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            original_query = user_messages[-1].content if user_messages else ""
            
            print(f"🎯 Generating recommendations: User ID={user_id}, Stock Codes={stock_codes}, Products={specific_products}")
            
            if user_id and self.df_data is not None:
                final_stock_codes = []
                
                # Handle stock codes directly provided
                if stock_codes:
                    final_stock_codes.extend(stock_codes)
                
                # Handle product descriptions - convert to stock codes
                if specific_products and not stock_codes:
                    for product_desc in specific_products:
                        stock_code = self._get_stock_code_from_description(product_desc)
                        if stock_code:
                            final_stock_codes.append(stock_code)
                
                # If no specific products/codes, get from user's history
                if not final_stock_codes:
                    user_purchases = self.df_data[self.df_data['CustomerID'] == user_id]
                    if len(user_purchases) > 0:
                        final_stock_codes = user_purchases['StockCode'].unique()[:3].tolist()
                
                if final_stock_codes:
                    print(f"📦 Using stock codes: {final_stock_codes}")
                    
                    # Run both recommendation algorithms
                    collaborative_results = self._collaborative_filtering(user_id, final_stock_codes, top_n=10)
                    content_based_results = self._content_based_filtering(user_id, final_stock_codes, recommendations_per_stock=3)
                    
                    # Rerank using LangGraph agent
                    final_recommendations = self._rerank_recommendations(collaborative_results, content_based_results, top_n=5)
                    
                    # Generate response
                    if final_recommendations:
                        response = self._format_langgraph_recommendations(final_recommendations, user_id, original_query)
                    else:
                        response = f"I couldn't find suitable recommendations for customer {user_id} at the moment. Please try with different preferences."
                    
                    state["messages"].append(AIMessage(content=response))
                    state["conversation_stage"] = "recommended"
                else:
                    # No purchase history or specific products
                    fallback_msg = f"""
I found customer {user_id}, but couldn't determine specific products for recommendations. 

Please specify:
• Specific product names or descriptions you're interested in
• Product stock codes you'd like similar items for
• Or let me know your general preferences

What type of products interest you most?
                    """
                    state["messages"].append(AIMessage(content=fallback_msg))
                    state["conversation_stage"] = "fallback"
            else:
                response = "Please provide your customer ID for personalized recommendations."
                state["messages"].append(AIMessage(content=response))
                state["conversation_stage"] = "needs_customer_id"
                
        except Exception as e:
            error_msg = f"I encountered an issue generating recommendations: {str(e)}. Let me try a different approach."
            state["messages"].append(AIMessage(content=error_msg))
            state["conversation_stage"] = "error"
            print(f"❌ Recommendation error: {e}")
        
        return state

    def _format_langgraph_recommendations(self, recommendations: List[Dict], user_id: int, original_query: str) -> str:
        """Format the LangGraph recommendations for display."""
        if not recommendations:
            return f"I couldn't find suitable recommendations for customer {user_id} at the moment."

        header = f"🎯 **Personalized Recommendations for Customer {user_id}:**\n\n"
        
        formatted_recs = []
        for i, rec in enumerate(recommendations, 1):
            formatted_recs.append(
                f"{i}. **{rec['Description']}** (Stock: {rec['Stock Code']})\n"
                f"   💰 Price: ${rec['Unit Price']:.2f}\n"
                # f"   📊 Source: {rec['Source']}\n"
                # f"   📈 Popularity: {rec['Popularity']}\n"
            )
        
        footer = "\n💡 These recommendations are based on your purchase history and similar customer preferences using advanced AI algorithms!"
        
        return header + "\n".join(formatted_recs) + footer

    def load_data(self):
        """Load and prepare recommendation data."""
        try:
            df = self.data_processor.load_data_from_sqlite()
            if df is not None:
                self.df_data = self.data_processor.clean_data(df)
                # self.crew_agents.set_dataframe(self.df_data)
                print(f"✅ Loaded {len(self.df_data)} transaction records")
            else:
                print("❌ Failed to load data")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def _execute_db_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute database query and return results."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def _get_comprehensive_customer_data(self, customer_id: int) -> Dict:
        """Get comprehensive customer data for RAG context."""
        
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

    def _format_customer_context(self, data: Dict) -> str:
        """Format customer data as context for LLM."""
        customer = data["customer_info"]
        summary = data["transaction_summary"]
        purchases = data["recent_purchases"]
        categories = data["top_categories"]
        
        context = f"""
Customer Information:
- Customer ID: {customer['CustomerID']}
- Location: {customer['Country']} - {customer['District']}
- Segment: {customer['Segment']}
- Total Lifetime Spending: ${customer['Customer_TotalSpending']:.2f}

Transaction Summary:
- Total Orders: {summary.get('total_orders', 0)}
- Total Items Purchased: {summary.get('total_items', 0)}
- Total Quantity: {summary.get('total_quantity', 0)}
- Average Item Price: ${summary.get('avg_item_price', 0):.2f}
- First Purchase: {summary.get('first_purchase', 'N/A')}
- Last Purchase: {summary.get('last_purchase', 'N/A')}
- Unique Products: {summary.get('unique_products', 0)}

Recent Purchases (Last 10):
"""
        
        for i, purchase in enumerate(purchases[:10], 1):
            context += f"{i}. {purchase['Description']} (Stock: {purchase['StockCode']}) - ${purchase['UnitPrice']:.2f} x {purchase['Quantity']} on {purchase['InvoiceDate']}\n"
        
        context += "\nTop Product Categories:\n"
        for i, cat in enumerate(categories, 1):
            context += f"{i}. {cat['category']}: {cat['purchase_count']} items, ${cat['total_spent']:.2f} spent\n"
        
        return context

    def _generate_intelligent_customer_response(self, query: str, data: Dict, context: str) -> str:
        """Generate intelligent response using LLM with customer context."""
        
        prompt = f"""
You are an expert customer service analyst. Based on the customer data provided, answer the user's specific question about this customer.

User Question: "{query}"

Customer Data Context:
{context}

Instructions:
1. Answer the specific question asked by the user
2. Be precise and direct
3. Use the exact data from the context
4. Include relevant insights when appropriate
5. Format the response clearly with emojis and structure
6. If asking about segment, just provide the segment value clearly
7. If asking about purchases, provide the requested purchase details
8. If asking about behavior, provide analytical insights
9. Always use the correct customer ID from the context

Generate a helpful, accurate response based on the specific question asked.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"❌ Error generating intelligent response: {e}")
            # Fallback to basic response
            customer = data["customer_info"]
            return f"📊 Customer {customer['CustomerID']} - Segment: {customer['Segment']}"
        
    def _get_customer_recent_purchases(self, customer_id: int, limit: int = 3) -> List[Dict]:
        """Get customer's recent purchases."""
        query = """
        SELECT t.StockCode_id as StockCode, p.Description, t.UnitPrice, t.Quantity, t.InvoiceDate
        FROM recommendations_fact_transactions t
        JOIN recommendations_dim_products p ON t.StockCode_id = p.StockCode
        WHERE t.CustomerID_id = ?
        ORDER BY t.InvoiceDate DESC
        LIMIT ?
        """
        results = self._execute_db_query(query, (customer_id, limit))
        return results
    
    def _get_customer_behavior(self, customer_id: int) -> Dict:
        """Get customer behavior from database with fixed table and column names."""
        query = """
        SELECT c.CustomerID, c.Country, c.District, c.Customer_TotalSpending, c.Segment,
            COUNT(t.InvoiceNo) as total_transactions,
            COUNT(DISTINCT t.StockCode_id) as unique_products
        FROM recommendations_dim_customers c
        LEFT JOIN recommendations_fact_transactions t ON c.CustomerID = t.CustomerID_id
        WHERE c.CustomerID = ?
        GROUP BY c.CustomerID, c.Country, c.District, c.Customer_TotalSpending, c.Segment
        """
        results = self._execute_db_query(query, (customer_id,))
        return results[0] if results else {}

    def _get_product_info(self, product_query: str) -> List[Dict]:
        """Get product information from database with fixed table names."""
        # Try to match by stock code first
        stock_code_query = """
        SELECT p.StockCode, p.Description, p.Description_Categorize,
            COUNT(t.InvoiceNo) as transaction_count,
            SUM(t.Quantity) as total_quantity_sold,
            AVG(t.UnitPrice) as avg_price
        FROM recommendations_dim_products p
        LEFT JOIN recommendations_fact_transactions t ON p.StockCode = t.StockCode_id
        WHERE p.StockCode LIKE ?
        GROUP BY p.StockCode, p.Description, p.Description_Categorize
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

    def _get_stock_code_from_description(self, description: str) -> str:
        """Get stock code from product description."""
        query = """
        SELECT StockCode FROM recommendations_dim_products 
        WHERE Description LIKE ? 
        LIMIT 1
        """
        results = self._execute_db_query(query, (f"%{description}%",))
        return results[0]['StockCode'] if results else None

    def _extract_context_from_history(self, state: ConversationState) -> Dict[str, Any]:
        """Extract relevant context from conversation history."""
        history = state.get("conversation_history", [])
        context = {
            "mentioned_customer_ids": [],
            "last_customer_id": None,
            "mentioned_products": [],
            "query_types": []
        }
        
        for entry in history:
            if entry.get("customer_id"):
                context["mentioned_customer_ids"].append(entry["customer_id"])
                context["last_customer_id"] = entry["customer_id"]
            
            if entry.get("products"):
                context["mentioned_products"].extend(entry["products"])
            
            if entry.get("query_type"):
                context["query_types"].append(entry["query_type"])
        
        return context

    def _update_conversation_history(self, state: ConversationState, query: str, customer_id: int = None, query_type: str = None):
        """Update conversation history with current interaction."""
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        entry = {
            "query": query,
            "timestamp": pd.Timestamp.now().isoformat(),
            "customer_id": customer_id,
            "query_type": query_type
        }
        
        state["conversation_history"].append(entry)
        
        # Keep only last 10 interactions
        if len(state["conversation_history"]) > 10:
            state["conversation_history"] = state["conversation_history"][-10:]

    def _resolve_contextual_references(self, query: str, context: Dict[str, Any]) -> tuple[str, int]:
        """Resolve contextual references like 'same customer', 'last customer ID'."""
        resolved_query = query
        resolved_customer_id = None
        
        # Check for contextual references
        if any(phrase in query.lower() for phrase in ["same customer", "that customer", "this customer"]):
            if context.get("last_customer_id"):
                resolved_customer_id = context["last_customer_id"]
                resolved_query = query.replace("same customer", f"customer {resolved_customer_id}")
                resolved_query = resolved_query.replace("that customer", f"customer {resolved_customer_id}")
                resolved_query = resolved_query.replace("this customer", f"customer {resolved_customer_id}")
        
        elif any(phrase in query.lower() for phrase in ["last customer", "previous customer"]):
            if context.get("last_customer_id"):
                resolved_customer_id = context["last_customer_id"]
                resolved_query = query.replace("last customer", f"customer {resolved_customer_id}")
                resolved_query = resolved_query.replace("previous customer", f"customer {resolved_customer_id}")
        
        return resolved_query, resolved_customer_id

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph conversation flow."""
        
        # Define the graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes (conversation stages)
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("understand_query", self._understand_query_node)
        workflow.add_node("extract_info", self._extract_info_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
        workflow.add_node("handle_customer_lookup", self._handle_customer_lookup_node)
        workflow.add_node("handle_product_inquiry", self._handle_product_inquiry_node)
        workflow.add_node("handle_general_question", self._handle_general_question_node)
        workflow.add_node("follow_up", self._follow_up_node)
        workflow.add_node("clarify", self._clarify_node)
        
        # Define the conversation flow
        workflow.set_entry_point("greeting")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "greeting",
            self._route_after_greeting,
            {
                "understand": "understand_query",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "understand_query",
            self._route_after_understanding,
            {
                "extract_info": "extract_info",
                "customer_lookup": "handle_customer_lookup",
                "product_inquiry": "handle_product_inquiry",
                "general_question": "handle_general_question",
                "clarify": "clarify",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "extract_info",
            self._route_after_extraction,
            {
                "recommend": "generate_recommendations",
                "clarify": "clarify",
            }
        )
        
        workflow.add_conditional_edges(
            "generate_recommendations",
            self._route_after_recommendations,
            {
                # "follow_up": "follow_up",
                "clarify": "clarify",
                "end": END
            }
        )
        
        # Add edges for new nodes - Fixed to END directly
        workflow.add_edge("handle_customer_lookup", END)
        workflow.add_edge("handle_product_inquiry", END)
        workflow.add_edge("handle_general_question", END)
        
        workflow.add_conditional_edges(
            "follow_up",
            self._route_after_follow_up,
            {
                "recommend": "generate_recommendations",
                "understand": "understand_query",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "clarify",
            self._route_after_clarification,
            {
                "understand": "understand_query",
                "extract_info": "extract_info",
                "recommend": "generate_recommendations",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.memory)
    
    # Node implementations
    def _greeting_node(self, state: ConversationState) -> ConversationState:
        """Handle initial greeting and introduction."""
        if not state["messages"] or state["conversation_stage"] == "greeting":
            greeting_msg = AIMessage(content="""
👋 Hello! I'm your AI-powered recommendation assistant. I can help you discover products based on your preferences and purchase history.

Just tell me what you're looking for in natural language! For example:
• "I need recommendations for customer 17850"
• "Show me products similar to what I bought before"  
• "I'm looking for gifts under $50"
• "What's popular in home decor?"
• "I want the last 3 purchased products for customer 12603"
• "What is the segment of customer 12357?"

If you have a customer ID, feel free to mention it for personalized recommendations!
            """.strip())
            
            state["messages"].append(greeting_msg)
            state["conversation_stage"] = "understanding"
            
        return state
    def _understand_query_node(self, state: ConversationState) -> ConversationState:
        """Analyze user query to understand their intent and needs using natural language processing."""
        if not state["messages"]:
            return state
            
        # Get the last user message
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state
            
        last_user_message = user_messages[-1].content
        print(f"🔍 Analyzing query: '{last_user_message[:50]}...'")

        # Extract context from conversation messages
        conversation_context = self._extract_context_from_messages(state)  # ✅ Uses built-in messages
        print(f"🐛 DEBUG - Extracted context: {conversation_context}")  # ADD THIS LINE
        # Resolve contextual references
        resolved_message, resolved_customer_id = self._resolve_contextual_references(
            last_user_message, conversation_context
        )
        if resolved_customer_id and not state.get("user_id"):
            state["user_id"] = resolved_customer_id
            print(f"🔄 Applied context customer ID: {resolved_customer_id}")
        
        # Also apply context customer ID if available
        if not state.get("user_id") and conversation_context.get("last_customer_id"):
            state["user_id"] = conversation_context["last_customer_id"]
            print(f"🔄 Applied context customer ID from history: {conversation_context['last_customer_id']}")
        
        print(f"🔍 Original query: '{last_user_message[:50]}...'")
        if resolved_message != last_user_message:
            state["user_id"] = resolved_customer_id
            print(f"🔄 Resolved query: '{resolved_message[:50]}...'")
        
        # Use resolved message for analysis
        analysis_message = resolved_message
        
        # Enhanced prompt for better intent classification
        # FIXED: Properly escape the JSON template to avoid f-string format errors
        analysis_prompt = f"""
Analyze this user message for an e-commerce recommendation system and classify the intent accurately:

User message: "{analysis_message}"

Conversation Context:
- Last mentioned customer ID: {conversation_context.get('last_customer_id', 'None')}
- Previously mentioned customer IDs: {conversation_context.get('mentioned_customer_ids', [])}

IMPORTANT CONTEXT RULES:
- If the user says "explain", "tell me more", "summarize it", "elaborate" and there's a last_customer_id, treat as "customer_lookup"
- If the user uses vague language but there's conversation context, infer the intent from context
- For contextual follow-ups, use the customer_id from context

Classification rules:
- "customer_lookup": User wants customer information, purchase history, or analysis
- "recommendation_request": User wants product recommendations
- "product_inquiry": User asks about specific products
- "general_question": User asks unrelated questions (ONLY if no context exists)

Please extract and return in JSON format. ANALYZE THE MESSAGE and determine the appropriate values:
{{
    "intent": "determine based on the message and context",
    "customer_id": "use context customer_id if the message refers to previous context, otherwise extract from message",
    "resolved_customer_id": {conversation_context.get('last_customer_id', 'null')},
    "product_preferences": {{
        "categories": ["list any mentioned categories"],
        "price_range": "any mentioned price range",
        "keywords": ["extract keywords from message"],
        "occasion": "determine from context"
    }},
    "confidence": 0.9,
    "query_type": "determine based on intent and message"
}}

Use conversation context to resolve ambiguous requests like "explain", "summarize it", etc.
"""
        
        try:
            analysis = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis_text = analysis.content
            
            # Try to extract JSON from the response
            try:
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = analysis_text[start_idx:end_idx]
                    extracted_info = json.loads(json_str)
                    state["extracted_info"] = extracted_info
                    
                    # Update state based on extracted info
                    customer_id = extracted_info.get("customer_id") or extracted_info.get("resolved_customer_id")
                    if customer_id:
                        try:
                            state["user_id"] = int(customer_id)
                        except (ValueError, TypeError):
                            pass
                    
                    # Also use resolved customer ID if available
                    if resolved_customer_id and not state.get("user_id"):
                        state["user_id"] = resolved_customer_id
                    
                    # Also try regex extraction as backup
                    if not state.get("user_id"):
                        user_id_match = re.search(r'\b(\d{5,})\b', last_user_message)
                        if user_id_match:
                            state["user_id"] = int(user_id_match.group(1))
                    
                    state["user_preferences"] = extracted_info.get("product_preferences", {})
                    state["context"]["intent"] = extracted_info.get("intent", "general_question")
                    state["context"]["analysis"] = extracted_info
                    state["conversation_stage"] = "analyzed"
                    
                    print(f"📊 Extracted: Intent={extracted_info.get('intent')}, Customer ID={extracted_info.get('customer_id')}")
                    
            except json.JSONDecodeError:
                print("⚠️ Could not parse JSON from analysis, using fallback extraction")
                # Fallback: simple regex extraction
                user_id_match = re.search(r'\b(\d{5,})\b', last_user_message)
                if user_id_match:
                    state["user_id"] = int(user_id_match.group(1))
                
                state["conversation_stage"] = "needs_extraction"
                
        except Exception as e:
            print(f"❌ Error in query analysis: {e}")
            state["conversation_stage"] = "clarify"
        
        return state
    def _extract_info_node(self, state: ConversationState) -> ConversationState:
        """Enhanced information extraction and validation."""
        extracted = state.get("extracted_info", {})
        
        # Check if we have sufficient information for recommendations
        has_customer_id = state.get("user_id") is not None
        has_preferences = bool(state.get("user_preferences", {}).get("keywords"))
        intent = extracted.get("intent", "general_question")
        confidence = extracted.get("confidence", 0.0)
        
        print(f"📋 Info check: Customer ID={has_customer_id}, Preferences={has_preferences}, Intent={intent}, Confidence={confidence}")
        
        # Determine if we can proceed with recommendations
        if intent == "recommendation_request":
            if has_customer_id:
                state["conversation_stage"] = "ready_to_recommend"
            else:
                # Try to use context customer ID
                conversation_context = self._extract_context_from_messages(state)
                if conversation_context.get("last_customer_id"):
                    state["user_id"] = conversation_context["last_customer_id"]
                    state["conversation_stage"] = "ready_to_recommend"
                    print(f"🔄 Using context customer ID: {conversation_context['last_customer_id']}")
                else:
                    state["conversation_stage"] = "needs_clarification"
        elif confidence < 0.7 or intent == "general_question":
            state["conversation_stage"] = "needs_clarification"
        else:
            state["conversation_stage"] = "needs_clarification"
        
        return state

    def _handle_customer_lookup_node(self, state: ConversationState) -> ConversationState:
        """Handle customer lookup queries using RAG approach with comprehensive data."""
        user_id = state.get("user_id")
        
        # Get the original user query
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        original_query = user_messages[-1].content if user_messages else ""
        
        print(f"🔍 Debug - User ID: {user_id}, Original Query: {original_query}")
        
        if not user_id:
            # Try to extract from the message
            if user_messages:
                user_id_match = re.search(r'\b(\d{5,})\b', user_messages[-1].content)
                if user_id_match:
                    user_id = int(user_id_match.group(1))
        
        if user_id:
            # Get comprehensive customer data
            customer_data = self._get_comprehensive_customer_data(user_id)
            
            if customer_data.get("customer_info"):
                # Create context for the LLM
                context = self._format_customer_context(customer_data)
                
                # Generate intelligent response using LLM
                response = self._generate_intelligent_customer_response(original_query, customer_data, context)
                
                print(f"🔍 Debug - Generated intelligent response length: {len(response)}")
            else:
                response = f"❌ Customer {user_id} not found in our database."
        else:
            response = "Please provide a valid customer ID for lookup."
        
        state["messages"].append(AIMessage(content=response))
        state["conversation_stage"] = "customer_analyzed"
        return state

    def _handle_product_inquiry_node(self, state: ConversationState) -> ConversationState:
        """Handle product inquiry queries."""
        extracted = state.get("extracted_info", {})
        specific_products = extracted.get("specific_products", [])
        keywords = extracted.get("product_preferences", {}).get("keywords", [])
        
        # Combine all search terms
        search_terms = specific_products + keywords
        
        if search_terms:
            all_products = []
            for term in search_terms[:3]:  # Limit to first 3 terms
                products = self._get_product_info(term)
                all_products.extend(products)
            
            if all_products:
                # Remove duplicates based on StockCode
                unique_products = {p['StockCode']: p for p in all_products}.values()
                limited_products = list(unique_products)[:5]  # Limit to 5 products
                
                response = "🔍 **Product Information:**\n\n"
                for i, product in enumerate(limited_products, 1):
                    response += f"{i}. **{product['Description']}** ({product['StockCode']})\n"
                    response += f"   📂 Category: {product.get('Description_Categorize', 'N/A')}\n"
                    response += f"   💰 Avg Price: ${product.get('avg_price', 0):.2f}\n"
                    response += f"   📊 Transactions: {product.get('transaction_count', 0)}\n"
                    response += f"   📦 Total Sold: {product.get('total_quantity_sold', 0) or 0}\n\n"
            else:
                response = f"❌ No products found matching: {', '.join(search_terms)}"
        else:
            response = "Please specify which products you'd like to know about."
        
        state["messages"].append(AIMessage(content=response))
        state["conversation_stage"] = "product_analyzed"
        return state
    
    def _handle_general_question_node(self, state: ConversationState) -> ConversationState:
        """Handle general questions outside e-commerce scope."""
        response = """
I'm specialized in e-commerce recommendations and product analysis. I can help you with:

• Product recommendations based on customer preferences
• Customer purchase history and behavior analysis  
• Product information and details
• Purchase trend analysis
• Customer segmentation and insights

Please ask me something related to our e-commerce platform!
        """
        state["messages"].append(AIMessage(content=response))
        state["conversation_stage"] = "out_of_scope"
        return state
    def _generate_intelligent_recommendation_response(self, query: str, crew_results, user_id: int, extracted_info: Dict) -> str:
        """Generate intelligent recommendation response using LLM with CrewAI context."""
        
        # Extract raw CrewAI output as context
        crew_context = ""
        if hasattr(crew_results, 'raw') and crew_results.raw:
            crew_context = crew_results.raw
        
        # Get customer context for better recommendations
        customer_data = self._get_comprehensive_customer_data(user_id)
        customer_context = ""
        if customer_data.get("customer_info"):
            customer = customer_data["customer_info"]
            recent_purchases = customer_data["recent_purchases"][:3]  # Last 3 purchases
            
            customer_context = f"""
    Customer Profile:
    - Customer ID: {customer['CustomerID']}
    - Segment: {customer['Segment']}
    - Location: {customer['Country']} - {customer['District']}
    - Total Spending: ${customer['Customer_TotalSpending']:.2f}

    Recent Purchase History:
    """
            for i, purchase in enumerate(recent_purchases, 1):
                customer_context += f"{i}. {purchase['Description']} - ${purchase['UnitPrice']:.2f} on {purchase['InvoiceDate']}\n"
        
        prompt = f"""
    You are an expert e-commerce recommendation specialist. Based on the user's query and the AI-generated recommendations, create a personalized response.

    User Query: "{query}"

    Customer Context:
    {customer_context}

    AI Recommendation Results:
    {crew_context}

    Instructions:
    1. Parse the recommendation data from the AI results
    2. Present the recommendations in a clear, engaging format
    3. Include relevant details like product names, prices, stock codes
    4. Add personalized insights based on customer profile
    5. Use emojis and proper formatting
    6. Explain WHY these products are recommended
    7. Make it conversational and helpful
    8. If the user asked for specific number of items, respect that

    Generate a comprehensive, personalized recommendation response.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"❌ Error generating intelligent recommendation response: {e}")
            # Fallback to parsing and basic formatting
            return self._fallback_recommendation_format(crew_results, user_id)

    def _fallback_recommendation_format(self, crew_results, user_id: int) -> str:
            """Fallback method to format recommendations if LLM fails."""
            try:
                # Try to parse JSON from CrewAI results
                if hasattr(crew_results, 'raw') and crew_results.raw:
                    result_text = crew_results.raw
                    
                    # Find JSON in the response
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = result_text[start_idx:end_idx]
                        parsed_results = json.loads(json_str)
                        recommendations = parsed_results.get('Top Recommendations', [])
                        
                        if recommendations:
                            response = f"🎯 **Personalized Recommendations for Customer {user_id}:**\n\n"
                            for i, rec in enumerate(recommendations, 1):
                                response += f"{i}. **{rec.get('Description', 'Unknown Product')}** (Stock: {rec.get('Stock Code', 'N/A')})\n"
                                response += f"   💰 Price: ${rec.get('Unit Price', 0):.2f}\n"
                                # response += f"   📊 Popularity Score: {rec.get('Popularity', 'N/A')}\n"
                                # response += f"   🔍 Source: {rec.get('Source', 'AI Analysis')}\n\n"
                            
                            return response + "💡 These recommendations are based on your purchase history and similar customer preferences!"
            except Exception as e:
                print(f"❌ Fallback formatting error: {e}")
            
            return f"I found some great recommendations for customer {user_id}, but had trouble formatting them. Please try again!"
            
    def _follow_up_node(self, state: ConversationState) -> ConversationState:
        """Handle follow-up questions and refinements."""
        follow_up_msg = """
How can I help you further? I can:

• Provide more recommendations for different customers
• Look up specific customer purchase history
• Search for detailed product information
• Analyze customer behavior patterns
• Check customer segments and profiles

What would you like to explore next?
        """
        
        state["messages"].append(AIMessage(content=follow_up_msg))
        state["conversation_stage"] = "following_up"
        
        return state
    
    def _clarify_node(self, state: ConversationState) -> ConversationState:
        """Ask for clarification when the user's request is unclear."""
        clarify_msg = """
I want to make sure I understand what you're looking for. Could you help me by sharing:

• Your customer ID (if you have one) for personalized suggestions or analysis
• Specific product names, descriptions, or stock codes you're interested in
• What type of information you need:
  - Product recommendations
  - Customer purchase history  
  - Product details and information
  - Customer behavior analysis
  - Customer segment information

The more details you provide, the better I can assist you!
        """
        
        state["messages"].append(AIMessage(content=clarify_msg))
        state["conversation_stage"] = "clarifying"
        
        return state
    
    # Routing functions
    def _route_after_greeting(self, state: ConversationState) -> str:
        if len(state["messages"]) > 1:
            return "understand"
        return "end"
    
    def _route_after_understanding(self, state: ConversationState) -> str:
        extracted = state.get("extracted_info", {})
        intent = extracted.get("intent", "general_question")
        
        if intent == "customer_lookup":
            return "customer_lookup"
        elif intent == "product_inquiry":
            return "product_inquiry"
        elif intent == "recommendation_request":
            return "extract_info"
        elif intent == "general_question":
            return "general_question"
        else:
            return "clarify"
    
    def _route_after_extraction(self, state: ConversationState) -> str:
        stage = state.get("conversation_stage", "")
        if stage == "ready_to_recommend":
            return "recommend"
        else:
            return "clarify"
    
    def _route_after_recommendations(self, state: ConversationState) -> str:
        stage = state.get("conversation_stage", "")
        if stage in ["recommended", "fallback", "needs_customer_id"]:
            return "end"  # Changed from "follow_up"
        return "clarify"
    
    def _route_after_follow_up(self, state: ConversationState) -> str:
        if state["messages"]:
            last_msg = state["messages"][-1].content.lower() if isinstance(state["messages"][-1], HumanMessage) else ""
            if any(word in last_msg for word in ["more", "other", "different", "another"]):
                return "recommend"
            elif any(word in last_msg for word in ["new", "something else", "change"]):
                return "understand"
        return "end"
    
    def _route_after_clarification(self, state: ConversationState) -> str:
        return "understand"
    
    # Helper methods
    def _parse_crew_results(self, results) -> List[Dict[str, Any]]:
        """Parse CrewAI results into standardized format with better error handling."""
        recommendations = []
        try:
            if hasattr(results, 'raw') and results.raw:
                result_text = results.raw
                print(f"🔍 Raw CrewAI result: {result_text[:200]}...")
                
                if isinstance(result_text, str):
                    # Try to find JSON in the response
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = result_text[start_idx:end_idx]
                        try:
                            parsed_results = json.loads(json_str)
                            raw_recommendations = parsed_results.get('recommendations', parsed_results.get('Top Recommendations', []))
                            
                            for rec in raw_recommendations:
                                recommendations.append({
                                    'stock_code': rec.get('Stock Code', rec.get('stock_code', '')),
                                    'description': rec.get('Description', rec.get('description', '')),
                                    'unit_price': rec.get('Unit Price', rec.get('unit_price', 0)),
                                    'confidence': rec.get('Confidence', rec.get('confidence', 0)),
                                    # 'source': rec.get('Source', 'CrewAI'),
                                    # 'popularity': rec.get('Popularity', rec.get('popularity', 'Medium'))
                                })
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON decode error: {e}")
                            # Fallback: try to extract structured data from text
                            recommendations = self._parse_text_recommendations(result_text)
                    else:
                        # No JSON found, try text parsing
                        recommendations = self._parse_text_recommendations(result_text)
                        
        except Exception as e:
            print(f"❌ Error parsing CrewAI results: {e}")
        
        return recommendations
    
    def _parse_text_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Fallback method to parse recommendations from plain text."""
        recommendations = []
        try:
            # Look for patterns like "Stock Code: XXX, Description: YYY, Price: ZZZ"
            lines = text.split('\n')
            current_rec = {}
            
            for line in lines:
                if 'stock code' in line.lower() or 'stockcode' in line.lower():
                    if current_rec:
                        recommendations.append(current_rec)
                        current_rec = {}
                    
                    # Extract stock code
                    stock_match = re.search(r'([A-Z0-9]+)', line)
                    if stock_match:
                        current_rec['stock_code'] = stock_match.group(1)
                
                elif 'description' in line.lower() and current_rec:
                    # Extract description
                    desc_match = re.search(r'description:?\s*(.+)', line, re.IGNORECASE)
                    if desc_match:
                        current_rec['description'] = desc_match.group(1).strip()
                
                elif 'price' in line.lower() and current_rec:
                    # Extract price
                    price_match = re.search(r'[\$£]?([\d.]+)', line)
                    if price_match:
                        current_rec['unit_price'] = float(price_match.group(1))
            
            if current_rec:
                recommendations.append(current_rec)
                
        except Exception as e:
            print(f"⚠️ Error in text parsing: {e}")
        
        return recommendations
    
    def _format_recommendations(self, recommendations: List[Dict], user_id: int = None, extracted_info: Dict = None) -> str:
        """Format recommendations for user display with enhanced context and styling."""
        if not recommendations:
            return "I couldn't find specific recommendations at the moment. Would you like me to try a different approach or search for different products?"
        
        # Create contextual header based on extracted information
        context = extracted_info.get("context", "") if extracted_info else ""
        occasion = extracted_info.get("product_preferences", {}).get("occasion", "") if extracted_info else ""
        
        if user_id:
            if occasion == "gift":
                header = f"🎁 **Gift Recommendations for Customer {user_id}:**\n\n"
            elif context == "business":
                header = f"💼 **Business Recommendations for Customer {user_id}:**\n\n"
            else:
                header = f"🎯 **Personalized Recommendations for Customer {user_id}:**\n\n"
        else:
            header = "🛍️ **Product Recommendations:**\n\n"
        
        # Format each recommendation with enhanced details
        formatted_recs = []
        for i, rec in enumerate(recommendations, 1):
            confidence = rec.get('confidence', 0)
            popularity = rec.get('popularity', 'Medium')
            source = rec.get('source', 'AI Analysis')
            
            formatted_recs.append(
                f"{i}. **{rec.get('description', 'Unknown Product')}** ({rec.get('stock_code', 'N/A')})\n"
                f"   💰 Price: ${rec.get('unit_price', 0):.2f}\n"
                f"   🎯 Match Score: {confidence:.1f}/10\n"
                # f"   📈 Popularity: {popularity}\n"
                # f"   🔍 Source: {source}\n"
            )
        
        footer = "\n💡 Would you like more details about any of these products, or shall I find different recommendations based on other preferences?"
        
        return header + "\n".join(formatted_recs) + footer
    
    # Main interaction methods
    def chat(self, message: str, user_id: str = "default_user") -> str:
        """Main chat interface with natural language processing."""
        # Create thread config - this is the key for persistence
        config = {"configurable": {"thread_id": "1"}}
        
        # DEBUG: Print thread and message info
        print(f"🧵 Thread ID: {user_id}")
        print(f"📝 Message: {message}")
        
        # Get current state - this should retrieve existing conversation
        try:
            current_state = self.graph.get_state(config)
            print(f"🔍 Current state exists: {current_state is not None}")
            print(f"🔍 Current state values: {current_state.values is not None if current_state else False}")
            
            if current_state and current_state.values:
                # ✅ Use existing state with conversation history
                state = current_state.values
                existing_messages = state.get('messages', [])
                print(f"🔄 Found existing state with {len(existing_messages)} messages")
                for i, msg in enumerate(existing_messages):
                    msg_type = "USER" if isinstance(msg, HumanMessage) else "AI"
                    print(f"  Message {i}: {msg_type}: {msg.content[:50]}...")
            else:
                # Only create new state for very first conversation
                print(f"🆕 Creating new conversation state")
                state = ConversationState(
                    messages=[],
                    user_id=None,
                    context={},
                    last_recommendations=[],
                    conversation_stage="greeting",
                    user_preferences={},
                    extracted_info={},
                    conversation_history=[]
                )
        except Exception as e:
            print(f"❌ Error getting state: {e}")
            # Fallback to new state
            state = ConversationState(
                messages=[],
                user_id=None,
                context={},
                last_recommendations=[],
                conversation_stage="greeting", 
                user_preferences={},
                extracted_info={},
                conversation_history=[]
            )
        
        # Add NEW user message to existing state
        if message.strip():
            state["messages"].append(HumanMessage(content=message))
            print(f"📝 Added user message. Total messages now: {len(state['messages'])}")
        
        # Run the graph with the state - LangGraph will automatically checkpoint
        try:
            result = self.graph.invoke(state, config)
            
            # Extract the response
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                response = ai_messages[-1].content
                print(f"🤖 Generated response length: {len(response)}")
                return response
            else:
                return "I'm here to help with recommendations! What are you looking for?"
                
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return "I encountered an issue. Could you please rephrase your request?"
        
    # def chat_stream(self, message: str, user_id: str = "default_user"):
    #     for chunk in self.llm.stream(
    #         [HumanMessage(content=message)],
    #         stream_mode="updates"
    #     ):
    #         yield chunk  

# Django Integration
def create_chatbot_view(request):
    """Django view for the chatbot interface."""
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        
        # Use singleton chatbot instance to preserve memory
        global _chatbot_instance
        if _chatbot_instance is None:
            print("🔧 Initializing new chatbot instance...")
            _chatbot_instance = RecommendationChatbot()
        
        response = _chatbot_instance.chat(message, user_id)
        
        return JsonResponse({
            'response': response,
            'status': 'success'
        })
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)