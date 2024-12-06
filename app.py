import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TherapistAssistant:
    def __init__(self, model_path: str):
        """Initialize the Therapist Assistant."""
        logger.info("Initializing Therapist Assistant...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model in half precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load PEFT adapter if it exists
        adapter_path = os.path.join(model_path, "adapter_model")
        if os.path.exists(adapter_path):
            logger.info("Loading PEFT adapter...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
        logger.info("Model loaded successfully!")
        
        # Define safety patterns
        self.safety_patterns = [
            r'suicid\w*',
            r'kill\w*',
            r'harm\w*',
            r'hurt\w*',
            r'die\w*',
            r'death\w*'
        ]
        
        # Crisis response template
        self.crisis_response = """I notice that you might be going through a difficult time. While I'm here to listen, I strongly encourage you to:

1. Contact emergency services (911 in the US) if you're in immediate danger
2. Call the National Crisis Hotline (988) to speak with a trained counselor
3. Reach out to a mental health professional or therapist
4. Talk to someone you trust about what you're experiencing

Your life is valuable, and help is available. Would you like information about mental health resources in your area?"""

    def check_safety(self, text: str) -> tuple[bool, str]:
        """Check if the input contains concerning content."""
        text_lower = text.lower()
        for pattern in self.safety_patterns:
            if re.search(pattern, text_lower):
                return False, self.crisis_response
        return True, ""

    def generate_response(
        self,
        instruction: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """Generate a response to the user's input."""
        # Input validation
        if len(instruction.strip()) < 3:
            return "Please provide a more detailed message so I can help you better."
        
        # Safety check
        is_safe, crisis_response = self.check_safety(instruction)
        if not is_safe:
            return crisis_response
        
        # Format input
        prompt = f"### Instruction: {instruction}\n\n### Response:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        response = response.split("### Response:")[-1].strip()
        
        # Quality check
        if len(response) < 10:
            return "I apologize, but I couldn't generate a proper response. Could you rephrase your message?"
        
        return response

def create_interface(model_path: str):
    """Create the Gradio interface."""
    # Initialize model
    assistant = TherapistAssistant(model_path)
    
    # Define CSS
    css = """
    .gradio-container {max-width: 800px !important}
    .chat-message {padding: 15px; border-radius: 10px; margin-bottom: 10px}
    .user-message {background-color: #e3f2fd}
    .assistant-message {background-color: #f5f5f5}
    .disclaimer {color: #d32f2f; font-weight: bold; padding: 10px; border: 1px solid #d32f2f; border-radius: 5px; margin: 10px 0}
    .controls {padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin: 10px 0}
    """
    
    # Create interface
    with gr.Blocks(css=css) as interface:
        gr.Markdown("# 🧠 AI Therapist Assistant")
        
        with gr.Box(class_name="disclaimer"):
            gr.Markdown("""
            ⚠️ **Important Disclaimer**
            - This is an AI assistant and NOT a replacement for professional therapy
            - In case of emergency, call your local emergency services or crisis hotline
            - All conversations are processed locally and are not stored
            """)
        
        chatbot = gr.Chatbot(label="Conversation")
        msg = gr.Textbox(
            label="Type your message here...",
            placeholder="How are you feeling today?",
            lines=2
        )
        
        with gr.Box(class_name="controls"):
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Response Creativity (Temperature)"
                )
                response_length = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=512,
                    step=50,
                    label="Maximum Response Length"
                )
            clear = gr.Button("Clear Conversation")
        
        def respond(message, history, temp, length):
            if not message:
                return "", history
            
            bot_message = assistant.generate_response(
                message,
                temperature=temp,
                max_length=length
            )
            
            history.append((message, bot_message))
            return "", history
        
        msg.submit(
            respond,
            [msg, chatbot, temperature, response_length],
            [msg, chatbot]
        )
        
        clear.click(lambda: None, None, chatbot, queue=False)
        
        gr.Markdown("""
        ### 📋 Usage Tips
        1. Be specific about your feelings and concerns
        2. Take your time to express yourself
        3. Remember this is an AI assistant - seek professional help for serious issues
        4. Use the controls above to adjust response style
        
        ### 🆘 Emergency Resources
        - Emergency Services: 911 (US)
        - National Crisis Hotline: 988 (US)
        - National Suicide Prevention Lifeline: 1-800-273-8255
        """)
    
    return interface

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run the AI Therapist Assistant interface')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--share', action='store_true', help='Create a shareable link')
    args = parser.parse_args()
    
    # Create and launch interface
    interface = create_interface(args.model_path)
    interface.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=8080,
        auth=None
    )

if __name__ == "__main__":
    main() 