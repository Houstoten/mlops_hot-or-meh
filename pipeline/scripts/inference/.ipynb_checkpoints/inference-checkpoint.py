import os
import json

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def load_model_for_inference(model_path):
    model = BERTClassifier('bert-base-uncased', 2)
    
    print(f"Extracting model from path: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model


def model_fn(model_dir):
    print("model init start ", model_dir, " ", os.listdir(model_dir))

    model_path = os.path.join(model_dir, 'model.pth')
    model = load_model_for_inference(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("model init end")

    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        print("received body: ", data)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encoding = tokenizer(data['text'], return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        return encoding
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_data = input_data.to(device)
        outputs = model(input_ids=input_data['input_ids'].to(device), attention_mask=input_data['attention_mask'].to(device))
        _, preds = torch.max(outputs, dim=1)

    return preds.item()


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        print("Prediction: ", prediction)
        return json.dumps({'result': "Hot" if prediction == 1 else "Meh"})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
