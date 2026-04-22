Model link download : https://drive.google.com/drive/folders/1t3iBbFy15aJZUt9wqWOO7Nt8UAOnNyBo?usp=drive_link

Dataset : https://drive.google.com/drive/folders/1y_xth4fYxDL4z4iGuXLFhaxdAXN2X1Uu?usp=sharing

# 🎮 Toxic Comment Classification System for Online Gaming Chat

A machine learning-based system that detects toxic behavior in real-time gaming chat using advanced Natural Language Processing (NLP) techniques. This project leverages a fine-tuned BERT model to understand gaming-specific language, slang, and contextual nuances.

---

## 📌 Overview

Online gaming communities often suffer from toxic behavior such as harassment, hate speech, and abusive language. Traditional moderation methods are either:
- ❌ Not scalable (manual moderation)
- ❌ Context-blind (keyword filtering)

This project introduces a **context-aware toxic comment classifier** designed specifically for gaming environments.

---

## 🚀 Features

- 🧠 **BERT-based classification model**
- 🎮 **Gaming-specific slang understanding** (e.g., "noob", "gg", "ff")
- ⚡ **Real-time prediction**
- 📊 **Probability-based toxicity scoring**
- 🌐 **Interactive web interface (Streamlit)**
- 📈 **Evaluation using Accuracy, Precision, Recall, F1-score**

---

## 🏗️ System Architecture

 User Input → Preprocessing → Tokenization → BERT Model → Classification Output

 
- **Frontend**: Streamlit Web Interface  
- **Backend**: Python + PyTorch  
- **Model**: Fine-tuned BERT (Hugging Face Transformers)

---

## 🧪 Model Details

- Base Model: `bert-base-uncased`
- Fine-tuned on:
  - Jigsaw Toxic Comment Dataset (~159k samples)
  - Gaming dataset (Dota 2 chat data)
- Multi-label classification:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate

---

## 📊 Performance Metrics

| Metric      | Description                          |
|------------|--------------------------------------|
| Accuracy   | Overall correctness                  |
| Precision  | Correct positive predictions         |
| Recall     | Coverage of actual toxic cases       |
| F1-Score   | Balance of precision and recall      |

---

## 🧰 Tech Stack

- **Language**: Python 3.10
- **ML Framework**: PyTorch
- **NLP Library**: Hugging Face Transformers
- **Data Processing**: Scikit-learn
- **Frontend**: Streamlit
- **Development Tools**: VS Code, Google Colab

---

## 📂 Project Structure

├── model/ # Trained model files
├── data/ # Dataset (optional / link only)
├── app.py # Streamlit application
├── fyp.ipynb # Model training notebook
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Installation

### 1. Clone the repository :

git clone https://github.com/your-username/toxic-gaming-classifier.git
cd toxic-gaming-classifier

### 2. Install dependencies :

pip install -r requirements.txt

### 3. Run the application :

streamlit run app.py

## 💻 Usage

Enter a gaming chat message
Click Analyze
View:
Toxic / Non-Toxic label
Confidence score

Example:

Input: "You are such a noob, uninstall the game"
Output: Toxic (92% confidence)

## 🧠 Key Innovation

Unlike generic toxicity detectors, this system:

Understands gaming context
Reduces false positives in competitive banter
Handles slang, abbreviations, and misspellings
