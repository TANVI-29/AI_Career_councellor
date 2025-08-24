from django.shortcuts import render

from app import preprocess_text
import numpy as np
import joblib 
# Create your views here.
def home(request):
    return render(request, 'app/ai.html')


from django.shortcuts import render
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from spellchecker import SpellChecker
import contractions
import re
import numpy as np
from django.http import HttpResponse

# -------------------- Load Model & Dataset --------------------
clf = joblib.load("model/career_clf.pkl")
le = joblib.load("model/label_encoder.pkl")
data = pd.read_csv("model/career_councellor.csv")
career_desc_dict = dict(zip(data['career'], data['description']))

# -------------------- Load Tokenizer & Embedding Model --------------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# -------------------- Preprocessing --------------------
spell = SpellChecker()
abbrev_dict = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "cs": "computer science",
    "ds": "data science",
    "dev": "developer",
    "prog": "programming",
    "bio":"biology",
    "maths":"mathematics"
}

def preprocess_text(text):
    if text is None:
        return ""
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in text.split()]
    text = " ".join(corrected_words)
    words = [abbrev_dict.get(w, w) for w in text.split()]
    text = " ".join(words)
    return text

# -------------------- Embedding Function --------------------
def embed_text(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:,0,:]
    return embeddings.numpy()

# -------------------- Career Prediction View --------------------
def career_page(request):
    recommendations = None  # default

    if request.method == "POST":
        # Get all 5 answers from frontend (match your JS keys)
        answers = []
        for i in range(1, 6):
            ans = request.POST.get(f"answer_{i}", "")
            answers.append(ans)

        # Combine all answers into one string
        combined_input = " ".join(answers)

        # Preprocess and embed
        preprocessed_input = preprocess_text(combined_input)
        embedding = embed_text([preprocessed_input])

        # Predict probabilities
        probs = clf.predict_proba(embedding)[0]

        # Get top 3 careers
        top_n = 3
        top_indices = np.argsort(probs)[::-1][:top_n]
        top_careers = le.inverse_transform(top_indices)
        top_probs = probs[top_indices]

        # Prepare recommendations
        recommendations = []
        for c, p in zip(top_careers, top_probs):
            recommendations.append({
                "career": c,
                "confidence": round(p*100, 2),
                "description": career_desc_dict.get(c, "Description not available")
            })

        # If this is an AJAX request, return HTML snippet
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            html = ""
            for rec in recommendations:
                html += f"""
                <div class="mb-6 p-4 bg-white rounded-lg shadow">
                    <strong class="text-lg">{rec['career']} ({rec['confidence']}% confident)</strong>
                    <p class="text-gray-700 mt-2">{rec['description']}</p>
                </div>
                """
            return HttpResponse(html)

    # For GET requests or normal POST page load
    return render(request, "app/questions.html", {"recommendations": recommendations})
