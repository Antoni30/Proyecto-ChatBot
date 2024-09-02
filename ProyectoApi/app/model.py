import torch
import re
from unidecode import unidecode
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

client = chromadb.PersistentClient(path="./BD_Preguntas")
collection = client.get_collection("PreguntasRespuestas")

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    return model,tokenizer

model,tokenizer = load_model()

def predict(features):
    query= clean_text(features).upper()
    query_vector = encode(query, tokenizer, model)
    results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
    )   
    print(results["distances"][0][0])
    return results['metadatas'][0][0]['Respuesta']

words_to_remove = ["CUAL", "ES", "EL", "QUE", "DE", "LA", "LAS", "SE", "A", "E", "I", "O", "U", "EN", "ESPE", "PARA", "LOS", "COMO", "SON"]

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[¿¡]', '', text) 
        text = re.sub(r'[?¡]', '', text) 
        text = unidecode(text)
        words = text.split()
        text = ' '.join([word for word in words if word.upper() not in words_to_remove])
    return text

def encode(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()