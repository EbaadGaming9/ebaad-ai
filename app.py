from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

MODEL = "deepseek-ai/deepseek-coder-1.3b"  # smaller, works better on free Railway
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")

def ebaad_ai(prompt):
    styled_prompt = f"You are Ebaad AI, a helpful, smart assistant. User asked: {prompt}"
    inputs = tokenizer(styled_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(
    fn=ebaad_ai, 
    inputs="text", 
    outputs="text", 
    title="💡 Ebaad AI",
    description="Your personal AI assistant"
).launch(server_name="0.0.0.0", server_port=8080)
