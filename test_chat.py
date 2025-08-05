# Save this as test_chat.py in your project root
import os
import sys
import django
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

# Configure Django settings BEFORE importing the chatbot
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_frontend.settings')
django.setup()

from src.chatbot.langgraph_chatbot import RecommendationChatbot

def main():
    print("🤖 Initializing chatbot...")
    try:
        chatbot = RecommendationChatbot()
        print("✅ Ready to chat!")
        
        print("\n" + "="*60)
        print("🤖 RECOMMENDATION CHATBOT")
        print("="*60)
        print("Try these queries:")
        print("• 'Hello'")
        print("• 'Show me customer 17850 behavior'")
        print("• 'Tell me about DOTCOM POSTAGE product'")
        print("• 'I need recommendations for customer 17850'")
        print("• 'What is the weather today?' (test general question)")
        print("• Type 'quit', 'exit', or 'bye' to end")
        print("-" * 60)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye', '']:
                print("👋 Goodbye!")
                break
            
            print("🤖 Assistant: ", end="")
            response = chatbot.chat(user_input, "test_user")
            print(response)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()