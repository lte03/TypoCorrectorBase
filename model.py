import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json

class SpellCorrectorDataset(Dataset):
    def __init__(self, wrong_words, correct_words):
        self.wrong_words = torch.LongTensor([encode_word(w, char_to_idx, max_length) for w in wrong_words])
        self.correct_words = torch.LongTensor([encode_word(c, char_to_idx, max_length) for c in correct_words])

    def __len__(self):
        return len(self.wrong_words)

    def __getitem__(self, idx):
        return self.wrong_words[idx], self.correct_words[idx]

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_output, decoder_output):
        score = self.V(torch.tanh(self.W1(encoder_output) + self.W2(decoder_output)))
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * encoder_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class SpellCorrectorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpellCorrectorModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.bilstm1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = AttentionLayer(hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        encoder_output, _ = self.bilstm1(x)
        encoder_output = self.dropout(encoder_output)
        encoder_output, _ = self.bilstm2(encoder_output)
        encoder_output = self.dropout(encoder_output)

        context_vector, attention_weights = self.attention(encoder_output, encoder_output)
        context_vector = context_vector.unsqueeze(1)
        context_vector = context_vector.repeat(1, encoder_output.size(1), 1)

        decoder_input = torch.cat([encoder_output, context_vector], dim=-1)
        output = self.fc(decoder_input)
        return torch.log_softmax(output, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("./lib/params_final.json") as f:
    params = json.load(f)

char_to_idx = params['char_to_idx']
idx_to_char = params['idx_to_char']
max_length = params['max_length']
EOS_TOKEN = params['eos_token']
hidden_dim = params['hidden_dim']
input_dim = len(char_to_idx) + 1
output_dim = len(char_to_idx) + 1

def encode_word(word, char_to_idx, max_length):
    encoded = [char_to_idx[char] for char in word] + [char_to_idx[EOS_TOKEN]]
    return encoded + [0] * (max_length - len(encoded))

def decode_sequence(sequence, idx_to_char):
    result = []
    for idx in sequence:
        if idx == 0:
            continue
        char = idx_to_char[str(idx)]
        if char == EOS_TOKEN:
            break
        result.append(char)
    return "".join(result)

def load_model_weights(model, file_path):
    try:
        model.load_state_dict(torch.load(file_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model weights loaded successfully from '{file_path}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")

def get_model():
    return SpellCorrectorModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

def correct_word(model,word):
    model.eval()
    with torch.no_grad():
        encoded = torch.LongTensor([encode_word(word, char_to_idx, max_length)]).to(device)
        output = model(encoded)
        predicted_indices = torch.argmax(output, dim=-1).cpu().numpy().squeeze()
        return decode_sequence(predicted_indices, idx_to_char)
