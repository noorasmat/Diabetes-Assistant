import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# ============================
# Load model and tokenizer
# ============================

@st.cache_resource
def load_model():
    # Load tokenizer from the original Hugging Face model to avoid SentencePiece issue
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=False)

    # Load your fine-tuned model from local directory
    model = AutoModelForCausalLM.from_pretrained("./tinyllama_diabetes_qlora")

    # Create HuggingFace pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        device=0 if torch.cuda.is_available() else -1,
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="TinyLlama Diabetes Assistant", page_icon="🧠")
st.title("🧠 TinyLlama Diabetes Assistant")
st.write("Ask questions related to diabetes education, treatment, and more.")

with st.form("query_form"):
    instruction = st.text_area("### Instruction", value="What is insulin resistance?")
    input_text = st.text_area("Additional context (optional)", value="")
    #input_text = st.text_area("### Input (optional)", value="")
    submit = st.form_submit_button("Generate Response")

if submit:
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    with st.spinner("Generating response..."):
        output = llm(prompt)
    st.markdown("### 📝 Response:")
    st.write(output)
