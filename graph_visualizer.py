# -*- coding: utf-8 -*-
"""
LangGraph Workflow Visualizer
Generates PNG and Mermaid diagrams of the recommendation chatbot workflow.
"""

import sys
import os

def visualize_langgraph():
    """Generate PNG and Mermaid visualizations of the LangGraph workflow."""
    
    # Add current directory to path
    sys.path.append('.')
    
    try:
        from src.chatbot.langgraph_chatbot import RecommendationChatbot
        
        print("🔧 Initializing recommendation chatbot...")
        chatbot = RecommendationChatbot()
        
        print("📊 Extracting graph structure...")
        graph_def = chatbot.graph.get_graph()
        
        # Generate PNG diagram
        print("🖼️ Generating PNG diagram...")
        png_data = graph_def.draw_mermaid_png()
        
        with open('langgraph_workflow.png', 'wb') as f:
            f.write(png_data)
        print("✅ PNG diagram saved to: langgraph_workflow.png")
        
        # Generate Mermaid code
        print("📝 Generating Mermaid code...")
        mermaid_code = graph_def.draw_mermaid()
        
        # Save Mermaid to markdown file
        with open('langgraph_workflow.md', 'w', encoding='utf-8') as f:
            f.write("# LangGraph Workflow Diagram\n\n")
            f.write("This diagram shows the conversation flow of the recommendation chatbot.\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("\n```\n\n")
            
            # Add node descriptions
            f.write("## Node Descriptions\n\n")
            f.write("- **__start__**: Entry point of the conversation\n")
            f.write("- **greeting**: Initial welcome and conversation start\n")
            f.write("- **understand_query**: Analyze user intent and query type\n")
            f.write("- **extract_info**: Extract relevant information from user input\n")
            f.write("- **generate_recommendations**: Create product recommendations\n")
            f.write("- **handle_customer_lookup**: Process customer data queries\n")
            f.write("- **handle_product_inquiry**: Handle product-specific questions\n")
            f.write("- **handle_general_question**: Process general support queries\n")
            f.write("- **follow_up**: Continue conversation flow\n")
            f.write("- **clarify**: Ask for clarification when needed\n")
            f.write("- **__end__**: Conversation termination\n")
        
        print("✅ Mermaid code saved to: langgraph_workflow.md")
        
        # Print basic statistics
        nodes = [str(node) for node in graph_def.nodes]
        edges = [(str(edge.source), str(edge.target)) for edge in graph_def.edges]
        
        print(f"\n📊 Graph Statistics:")
        print(f"   - Nodes: {len(nodes)}")
        print(f"   - Edges: {len(edges)}")
        print(f"   - Files: langgraph_workflow.png, langgraph_workflow.md")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = visualize_langgraph()
    if success:
        print("\n🎉 Visualization complete! Check the generated files.")
    else:
        print("\n💥 Visualization failed. Check the error messages above.")