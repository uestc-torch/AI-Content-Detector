from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('/home/icdm/CodeSpace/nlp_test/pretrained_best')
model = AutoModel.from_pretrained('/home/icdm/CodeSpace/nlp_test/pretrained_best').to(device)

# Function to convert text to embeddings with batching
def get_embeddings(text_list, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Generating Embeddings"):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Load dataset
dataset = pd.read_pickle('/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/dataset.pkl')

train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Extract text
train_text = train_data['Content'].tolist()
valid_text = valid_data['Content'].tolist()
test_text = test_data['Content'].tolist()

# Generate embeddings
train_embeddings = get_embeddings(train_text, batch_size=32)
valid_embeddings = get_embeddings(valid_text, batch_size=32)
test_embeddings = get_embeddings(test_text, batch_size=32)

# Add embeddings to DataFrame
train_data['mean_pooling_vec'] = train_embeddings.tolist()
valid_data['mean_pooling_vec'] = valid_embeddings.tolist()
test_data['mean_pooling_vec'] = test_embeddings.tolist()

# Save the datasets
train_data.to_pickle('/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/train.pkl')
valid_data.to_pickle('/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/valid.pkl')
test_data.to_pickle('/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/test.pkl')