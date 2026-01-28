import gradio as gr
import requests

def chat_with_model(message, history):
    """
    Send message to local Llama model and return response
    """
    # Ollama API endpoint
    url = "http://localhost:11434/api/chat"
    
    # Prepare messages in Ollama format
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})
    
    # Prepare request payload for Ollama
    payload = {
        "model": "llama3.2:latest",
        "messages": messages,
        "stream": False
    }
    
    try:
        # Send request to Ollama
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        bot_response = result.get("message", {}).get("content", "No response received")
        
        return bot_response
    
    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to Ollama. Make sure Ollama is running (try 'ollama serve')"
    except requests.exceptions.Timeout:
        return "❌ Error: Request timed out. The model might be taking too long to respond."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="🤖 Llama 3.2 Chatbot",
    description="Chat with your local Llama 3.2 model",
    examples=[
        "Hello! How are you?",
        "Explain quantum computing in simple terms",
        "Write a short poem about AI"
    ]
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )