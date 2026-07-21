import streamlit as st
import pickle
import numpy as np
import re
import ollama
from fuzzywuzzy import fuzz, process
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="MediSense AI",
    page_icon="🏥",
    layout="wide"
)

# ── Load artifacts ────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('medisense_preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('nlp_artifacts.pkl', 'rb') as f:
        nlp = pickle.load(f)
    return data, nlp

data, nlp_data = load_artifacts()
xgb            = data['xgb_model']
le             = data['label_encoder']
feature_names  = data['feature_names']
symptom_lookup = nlp_data['symptom_lookup']
symptom_phrases= nlp_data['symptom_phrases']
SYNONYMS       = nlp_data['synonyms']
GENERIC_SYMPTOMS = {'feeling ill', 'fatigue', 'weakness', 'ache all over'}

@st.cache_resource
def load_rag_from_pickle():
    try:
        with open('rag_knowledge.pkl', 'rb') as f:
            data = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.sidebar.success(f"✓ {len(data['documents'])} medical chunks loaded")
        return data, model
    except Exception as e:
        st.sidebar.warning("RAG not available")
        return None, None

rag_collection, embed_model = load_rag_from_pickle()


# ── Helper functions ──────────────────────────────────────
def generate_ngrams(text, max_n=5):
    words = text.split()
    ngrams = []
    for n in range(1, min(max_n, len(words)) + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
    return ngrams

def extract_symptoms_from_text(text, threshold=80):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    found_symptoms = set()
    working_text = text
    for syn, canonical in SYNONYMS.items():
        if syn in working_text:
            working_text += ' | ' + canonical
    sorted_phrases = sorted(symptom_phrases, key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in working_text:
            found_symptoms.add(symptom_lookup[phrase])
    ngrams = [g for g in generate_ngrams(working_text, max_n=4)
              if len(g) >= 4 and '|' not in g]
    for gram in ngrams:
        match, score = process.extractOne(gram, symptom_phrases, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            found_symptoms.add(symptom_lookup[match])
    redundant_pairs = [('back pain', 'low back pain')]
    for general, specific in redundant_pairs:
        gc = symptom_lookup.get(general)
        sc = symptom_lookup.get(specific)
        if gc in found_symptoms and sc in found_symptoms:
            found_symptoms.discard(gc)
    return list(found_symptoms)

def predict_disease_from_text(text, top_n=3):
    extracted = extract_symptoms_from_text(text)
    if not extracted:
        return None
    specific = [s for s in extracted if s not in GENERIC_SYMPTOMS]
    symptoms_for_model = specific if specific else extracted
    feature_vector = np.zeros(len(feature_names))
    for symptom in symptoms_for_model:
        if symptom in feature_names:
            feature_vector[feature_names.index(symptom)] = 1
    probs = xgb.predict_proba(feature_vector.reshape(1, -1))[0]
    top_indices = np.argsort(probs)[::-1][:top_n]
    return {
        'extracted': [s.replace('_', ' ') for s in extracted],
        'used':      [s.replace('_', ' ') for s in symptoms_for_model],
        'predictions': [
            {'disease': le.inverse_transform([i])[0],
             'confidence': round(float(probs[i]) * 100, 2)}
            for i in top_indices
        ]
    }

import re

def sanitize_input(text: str, max_length: int = 500) -> str:
    """
    Block prompt injection attempts with minimal overhead.
    Covers the most common attack patterns.
    """
    if not text or not text.strip():
        return ""

    # 1. Length cap — prevents context overflow attacks
    text = text[:max_length]

    # 2. Block common injection keywords/patterns
    injection_patterns = [
        r'ignore\s+(all\s+)?(previous|above|prior)\s+instructions?',
        r'forget\s+(everything|all|what)',
        r'you\s+are\s+now\s+a',
        r'act\s+as\s+(a\s+)?(?!patient)',   # allow "act as a patient" but not "act as a hacker"
        r'pretend\s+(to\s+be|you\s+are)',
        r'your\s+(new\s+)?role\s+is',
        r'system\s*:\s*',                    # fake system prompt injection
        r'<\s*system\s*>',                   # XML-style injection
        r'\[INST\]|\[\/INST\]',              # Llama instruction tags
        r'###\s*(instruction|system|prompt)',
        r'override\s+(safety|guidelines?|rules?)',
        r'reveal\s+(your\s+)?(prompt|instructions?|system)',
        r'what\s+(are\s+)?your\s+instructions?',
        r'(sudo|admin|root)\s*:',
        r'disregard\s+(your|all|the)',
        r'new\s+instructions?\s*:',
        r'translate\s+the\s+above',          # data exfiltration pattern
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "[BLOCKED: Invalid input detected. Please describe your symptoms only.]"

    # 3. Strip HTML/script tags (XSS via LLM output)
    text = re.sub(r'<[^>]+>', '', text)

    # 4. Normalize excessive whitespace/special chars
    text = re.sub(r'[\r\n]{3,}', '\n\n', text)
    text = re.sub(r'[^\x20-\x7E\n\u0900-\u097F]', '', text)  # keep ASCII + Devanagari

    return text.strip()

def retrieve_context(query, n_results=3):
    if not rag_collection or not embed_model:
        return ""
    try:
        query_embedding = embed_model.encode([query])
        doc_embeddings  = np.array(rag_collection['embeddings'])
        similarities    = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices     = np.argsort(similarities)[::-1][:n_results]
        return "\n\n---\n\n".join([rag_collection['documents'][i] for i in top_indices])
    except:
        return ""


# ── Groq client setup ──────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY") # or st.secrets.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)

def llm_chat(messages, system_prompt=None):
    """Drop-in replacement for ollama.chat()"""
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_report(text, result, language='English'):
    if not result or not result['predictions']:
        return "Could not generate report — no symptoms identified."

    symptoms_str    = ", ".join(result['used'])
    top_disease     = result['predictions'][0]['disease']
    predictions_str = "\n".join(
        [f"- {p['disease']}: {p['confidence']}%" for p in result['predictions']]
    )

    rag_context = retrieve_context(f"{top_disease} symptoms causes treatment", n_results=2)

    context_block = f"""
Use this verified WHO medical reference to ground your response:
---
{rag_context[:1000]}
---
Base your summary on this. Do not contradict it.""" if rag_context else ""

    prompt = f"""You are a careful, friendly medical triage assistant. Respond in {language}.

Patient described: "{text}"
Symptoms identified: {symptoms_str}
Possible conditions (NOT a confirmed diagnosis):
{predictions_str}
{context_block}
Write a warm, grounded summary in {language} (under 150 words):
1. Acknowledge symptoms understood
2. Briefly explain the top condition using reference material if available
3. List other possibilities as things to discuss with a doctor only
4. Strongly recommend seeing a healthcare professional
Never confirm a diagnosis. Use appropriate script for {language}."""

    return llm_chat([{"role": "user", "content": prompt}])


def chat_turn(history, user_msg, turn, specific_symptoms):
    symptoms_str = ", ".join(s.replace('_', ' ') for s in specific_symptoms) \
                   if specific_symptoms else "none clearly identified yet"

    if turn >= 3 and len(specific_symptoms) >= 1:
        readiness = (
            f"Symptoms identified: {symptoms_str}. "
            "Briefly confirm these back to the patient, then end with READY_FOR_PREDICTION on its own line."
        )
    elif turn >= 3:
        readiness = (
            f"Symptoms identified: {symptoms_str} — not enough yet. "
            "Ask a more targeted question about location, fever, swelling, or rash. "
            "Do NOT say READY_FOR_PREDICTION."
        )
    else:
        readiness = (
            f"Symptoms identified so far: {symptoms_str}. "
            "Ask ONE clarifying question. Do NOT say READY_FOR_PREDICTION yet."
        )

    system = f"""You are MediSense, a friendly medical triage chatbot.
You ONLY discuss medical symptoms and health conditions. 
If asked anything unrelated to health, respond: "I can only help with medical symptoms."
Gather symptoms one question at a time. Never diagnose yourself.
For urgent symptoms (severe chest pain, difficulty breathing) advise emergency care immediately.
{readiness}"""

    # Convert history format from streamlit to groq format
    groq_messages = []
    for msg in history:
        groq_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    groq_messages.append({"role": "user", "content": user_msg})

    return llm_chat(groq_messages, system_prompt=system)

# ── UI ────────────────────────────────────────────────────
st.title("🏥 MediSense AI")
with st.sidebar:
    st.markdown("## 🏥")
    st.markdown("### MediSense AI")
    st.markdown("---")
    st.markdown("**Knowledge Base**")
    if rag_collection:
        st.success(f"✓ {rag_collection.count()} medical chunks loaded")
        st.caption("Sources: WHO Fact Sheets, Clinical Guidelines")
    else:
        st.warning("RAG not loaded")
    st.markdown("---")
    st.warning("⚠️ Not a substitute for professional medical advice.")
st.caption("AI-powered symptom checker | For educational purposes only — not a medical diagnosis")

tab1, tab2 = st.tabs(["🔍 Symptom Checker", "💬 MediSense Chatbot"])

# ── TAB 1: Direct Symptom Checker ─────────────────────────
with tab1:
    st.subheader("Describe your symptoms")
    col1, col2 = st.columns([2, 1])

    with col1:
        symptom_text = st.text_area(
            "Type your symptoms in plain English",
            placeholder="e.g. I have a high fever, severe headache and joint pain...",
            height=120
        )
        language = st.selectbox("Report language", ["English", "Hindi", "Marathi"])

        if st.button("🔎 Analyze Symptoms", type="primary"):
            if not symptom_text.strip():
                st.warning("Please describe your symptoms first.")
            else:
                symptom_text = sanitize_input(symptom_text)
                if symptom_text.startswith("[BLOCKED"):
                    st.error("⚠️ " + symptom_text)
                else:
                    with st.spinner("Analyzing symptoms..."):
                        result = predict_disease_from_text(symptom_text)

                        if not result:
                            st.error("No recognizable symptoms found. Try describing more specifically.")
                        else:
                            st.success("Analysis complete!")

                    # Symptoms found
                    st.markdown("**Symptoms identified:**")
                    cols = st.columns(len(result['extracted']))
                    for i, s in enumerate(result['extracted']):
                        cols[i].info(s)

                    st.markdown("**Top predictions:**")
                    for p in result['predictions']:
                        st.markdown(f"**{p['disease'].title()}**")
                        st.progress(int(p['confidence']),
                                    text=f"{p['confidence']}% likelihood")

                    # GenAI report
                    st.markdown("---")
                    st.markdown("**📋 AI Health Summary:**")
                    with st.spinner(f"Generating {language} report..."):
                        report = generate_report(symptom_text, result, language)
                    st.info(report)
                    st.warning("⚠️ This is a screening tool only. Always consult a qualified doctor.")

    with col2:
        st.markdown("### How to use")
        st.markdown("""
        1. Type your symptoms in plain English
        2. Click **Analyze Symptoms**
        3. Review top possible conditions
        4. Read the AI-generated summary
        5. **Always follow up with a doctor**

        **Example inputs:**
        - *"fever, headache and joint pain"*
        - *"chest tightness and shortness of breath"*
        - *"itchy rash on arms with small blisters"*
        """)

# ── TAB 2: Chatbot ─────────────────────────────────────────
with tab2:
    st.subheader("Chat with MediSense")
    st.caption("Describe how you're feeling — MediSense will ask clarifying questions before generating a report.")

    # Session state for chat
    if 'chat_history'     not in st.session_state:
        st.session_state.chat_history     = []
    if 'chat_turn'        not in st.session_state:
        st.session_state.chat_turn        = 0
    if 'chat_done'        not in st.session_state:
        st.session_state.chat_done        = False
    if 'chat_full_text'   not in st.session_state:
        st.session_state.chat_full_text   = ""

    # Display conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role'],
                             avatar="👤" if msg['role'] == 'user' else "🤖"):
            st.write(msg['content'])

    # Input
    if not st.session_state.chat_done:
        user_input = st.chat_input("Describe your symptoms...")
        
        if user_input:
            user_input = sanitize_input(user_input)
            if user_input.startswith("[BLOCKED"):
                with st.chat_message("assistant", avatar="🤖"):
                    st.warning("Please describe your symptoms only. I can't process that input.")
            else:
            # Show user message
                with st.chat_message("user", avatar="👤"):
                    st.write(user_input)

                st.session_state.chat_turn += 1
                st.session_state.chat_full_text += " " + user_input

                # Extract symptoms from full conversation
                symptoms = extract_symptoms_from_text(st.session_state.chat_full_text)
                specific = [s for s in symptoms if s not in GENERIC_SYMPTOMS]

                # Get bot reply
                with st.spinner("MediSense is thinking..."):
                    reply = chat_turn(
                        st.session_state.chat_history,
                        user_input,
                        st.session_state.chat_turn,
                        specific
                    )

                display_reply = reply.replace("READY_FOR_PREDICTION", "").strip()
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(display_reply)

                st.session_state.chat_history.append({'role': 'user',    'content': user_input})
                st.session_state.chat_history.append({'role': 'assistant','content': display_reply})

                # Trigger prediction
                should_predict = (
                    "READY_FOR_PREDICTION" in reply or
                    (st.session_state.chat_turn >= 3 and len(specific) >= 1)
                )
                if should_predict:
                    with st.spinner("Generating your health report..."):
                        result = predict_disease_from_text(st.session_state.chat_full_text)
                        report = generate_report(st.session_state.chat_full_text, result)

                    st.markdown("---")
                    st.markdown("### 📋 MediSense Report")
                    st.markdown(f"**Symptoms used:** {', '.join(result['used'])}")

                    for p in result['predictions']:
                        st.markdown(f"**{p['disease'].title()}**")
                        st.progress(int(p['confidence']),
                                    text=f"{p['confidence']}% likelihood")

                    st.info(report)
                    st.warning("⚠️ Screening tool only. Always consult a qualified doctor.")
                    st.session_state.chat_done = True

    else:
        st.success("Consultation complete.")
        if st.button("🔄 Start new consultation"):
            for key in ['chat_history', 'chat_turn', 'chat_done', 'chat_full_text']:
                del st.session_state[key]
            st.rerun()