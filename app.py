import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Use a small DeepSeek model for Render free tier (big ones will fail)
MODEL = "deepseek-ai/deepseek-coder-1.3b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")

def ebaad_ai(prompt):
    styled_prompt = f"You are Ebaad AI, a helpful, smart assistant. User asked: {prompt}"
    inputs = tokenizer(styled_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=ebaad_ai,
    inputs="text",
    outputs="text",
    title="💡 Ebaad AI",
    description="Your personal AI assistant"
)

# 👇 IMPORTANT for Render: Use the PORT assigned by Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Render sets PORT automatically
    demo.launch(server_name="0.0.0.0", server_port=port)
