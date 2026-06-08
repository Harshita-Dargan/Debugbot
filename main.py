from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
import shap
import numpy as np

# --- Configuration ---
MODEL_PATH = "./codebert-defect-detector"
logging.basicConfig(level=logging.INFO)

# --- Pydantic Model for Input ---
class CodeInput(BaseModel):
    code_snippet: str

# --- Load Model & SHAP Explainer ---
logging.info(f"Loading model from {MODEL_PATH}...")
try:
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Create a text-classification pipeline
    classifier = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=-1, # Use -1 for CPU, 0 for first GPU
        return_all_scores=True # Important for SHAP
    )
    
    # Create a SHAP explainer for our pipeline
    # This wraps the classifier and allows us to get explanations
    explainer = shap.Explainer(classifier)

    logging.info("Model and SHAP explainer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model from {MODEL_PATH}. Did you run train.py first?")
    logging.error(e)
    classifier = None
    explainer = None

# --- Create FastAPI App ---
app = FastAPI()

# --- Add CORS Middleware ---
# This allows your frontend (running on file://) to talk to your backend (running on http://127.0.0.1:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---
@app.post("/analyze")
async def analyze_code(item: CodeInput):
    if classifier is None:
        return {"error": "Model is not loaded. Check server logs."}

    logging.info("Received analysis request...")
    try:
        # --- 1. Get Standard Prediction ---
        raw_results = classifier(item.code_snippet)[0]
        
        # Find the winning label
        prediction = max(raw_results, key=lambda x: x['score'])
        label = prediction['label']
        confidence = prediction['score']
        is_buggy = True if label == 'LABEL_1' else False
        
        # --- 2. Get SHAP Explanation ---
        logging.info("Running SHAP explainer...")
        
        # Get SHAP values
        shap_values = explainer([item.code_snippet])
        
        # We need the values for the *predicted* class
        # shap_values.base_values[0] will give the "base" score
        # shap_values.values[0] will give the influence scores
        # shap_values.data[0] will give the tokens
        
        # Get the index of the predicted label (0 for LABEL_0, 1 for LABEL_1)
        predicted_class_index = 1 if is_buggy else 0
        
        # Get the influence scores for the predicted class
        influence_scores = shap_values.values[0][:, predicted_class_index]
        tokens = shap_values.data[0]

        # Combine tokens and their scores
        # We also remove special tokens like '[CLS]' and '[SEP]'
        explanation = [
            (token, score) for token, score in zip(tokens, influence_scores)
            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
        ]
        
        logging.info("Analysis and explanation complete.")

        return {
            "label": label,
            "is_buggy": is_buggy,
            "confidence": confidence,
            "explanation": explanation # Send the new data to the frontend
        }
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return {"error": str(e)}