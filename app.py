import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Page Configuration ---
st.set_page_config(page_title="Gaming Chat Toxicity Classifier", page_icon="🎮")

# --- 2. Load the Model and Tokenizer ---
# We use @st.cache_resource so Streamlit only loads the model once, making the app much faster
@st.cache_resource
def load_model():
    # Make sure your downloaded model folder is named exactly this and is in the same folder as app.py
    model_path = "./toxic_gaming_model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

with st.spinner("Loading AI Model... Please wait."):
    tokenizer, model = load_model()
    
def preprocess_gaming_text(text):
    """Normalize common gaming slang before tokenization."""
    slang_map = {
        r'\bgg\b': 'good game',
        r'\bwp\b': 'well played',
        r'\bff\b': 'forfeit',
        r'\bnoob\b': 'newbie',
        r'\bn00b\b': 'newbie',
        r'\brekt\b': 'wrecked',
        r'\bafk\b': 'away from keyboard',
        r'\bkys\b': 'kill yourself',
        r'\bstfu\b': 'shut up',
        r'\btbh\b': 'to be honest',
        r'\bomg\b': 'oh my god',
        r'\bwtf\b': 'what the heck',
        r'\bffs\b': 'for goodness sake',
        r'\bu\b': 'you',
        r'\bur\b': 'your',
        r'\br\b': 'are',
    }
    import re
    text = text.lower().strip()
    for pattern, replacement in slang_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# --- 3. App UI Header ---
st.title("🎮 Toxic Gaming Chat Classifier")
st.write("Enter a chat message below to detect if it contains toxic behavior, general bad words, or gaming-specific insults.")

# --- Confidence Threshold Slider ---
st.sidebar.header("⚙️ Classifier Settings")
threshold = st.sidebar.slider(
    "Toxicity Detection Threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Lower = more sensitive (catches more toxic messages, but more false positives). Higher = more strict."
)

# --- 4. User Input ---
user_input = st.text_area("Chat Message:", placeholder="Type a gaming message here (e.g., 'gg well played' or 'trash team feed mid')...", height=100)

# --- 5. Prediction Logic ---
if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner("Analyzing with Hybrid AI..."):

            # Convert text to numbers for the AI
            cleaned_input = preprocess_gaming_text(user_input)
            inputs = tokenizer(cleaned_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # ⚠️ THE FIX: Delete 'token_type_ids' if the tokenizer created them!
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            
            # Get AI prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # THIS WAS THE MISSING LINE! Calculate the percentages:
                probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
            
            # --- 6. Display Results ---
            st.subheader("Analysis Results:")
            
            # Index 0 is Non-Toxic, Index 1 is Toxic
            clean_score = probabilities[0]
            toxic_score = probabilities[1] 
            
            # If the AI is more than 50% sure it is toxic
            if toxic_score > threshold:
                st.error(f"⚠️ **TOXIC GAMING CHAT DETECTED!**")
                st.write(f"The AI is **{toxic_score*100:.2f}%** confident this message is toxic.")
            else:
                st.success(f"✅ **CLEAN MESSAGE.**")
                st.write(f"The AI is **{clean_score*100:.2f}%** confident this message is safe/non-toxic.")
            
            # Visual Progress Bars
            st.write("---")
            st.write("### Confidence Breakdown:")
            
            st.write("**Non-Toxic Probability:**")
            st.progress(float(clean_score))
            
            st.write("**Toxic Probability:**")
            st.progress(float(toxic_score))
