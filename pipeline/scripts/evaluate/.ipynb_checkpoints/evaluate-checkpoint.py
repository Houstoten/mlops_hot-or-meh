import torch
import torch.nn as nn
import boto3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import BertTokenizer, BertModel
import numpy as np
import os
from io import StringIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import json
import tarfile
import pathlib
import mlflow
import logging
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

logging.getLogger("mlflow").setLevel(logging.DEBUG)

# model_path = '../models/bert_classifier.pth'
# model = BERTClassifier('bert-base-uncased', 2)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

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

def load_object_from_s3(key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='eu-north-1'
    )
    try:
        response = s3.get_object(Bucket='mlops-hot-or-meh', Key=key)
        serialized_model = response['Body'].read()
        return io.BytesIO(serialized_model)
        # return pickle.loads(serialized_model)
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Credentials not available", e)
        return None
    except Exception as e:
        print("Error occurred while fetching the model from S3", e)
        return None

def load_model_for_inference():
    # model_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models/bert_classifier.pth')
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print(f"Extracting model from path: {model_path}")

    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")

    model = BERTClassifier('bert-base-uncased', 2)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
        
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5

if __name__ == "__main__":

    print("Loading test input data...")
    test_features_data = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_data = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    model, tokenizer = load_model_for_inference()

    X_test = pd.read_csv(test_features_data)
    y_test = pd.read_csv(test_labels_data)

    # To be sure it won't fail
    X_test[X_test.columns[0]] = X_test[X_test.columns[0]].fillna('Empty Empty').astype(str)

    X_test_np = X_test[X_test.columns[0]].to_numpy()
    y_test_np = y_test[y_test.columns[0]].to_numpy()

    print("shapes: ", X_test_np.shape, y_test_np.shape)

    test_dataset = TextClassificationDataset(X_test_np, y_test_np, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)  # Use argmax for classification tasks

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for metric computation
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Compute metrics
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')  # Using weighted average for multiclass

    report_dict = {
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "r2_score": r2,
        "f1_score": f1,
    }

    print(f"Classification report:\n{report_dict}")
    
    print('Writing data to MLflow')
    mlflow.set_tracking_uri('arn:aws:sagemaker:us-east-2:471112582765:mlflow-tracking-server/mlops-hot-or-meh')
    
    experiment_name = 'model_stats'
    # mlflow.create_experiment(experiment_name, 's3://mlops-hot-or-meh2/mlflow-artifacts')
    mlflow.set_experiment(experiment_name)
    print(f'MLflow experiment: {experiment_name}')
    signature = mlflow.models.infer_signature(X_test_np, predictions)

    with mlflow.start_run(run_name=f'Run {type(model).__name__}') as run:
        print(f'Run id: {run.info.run_id}')

        report_dict["run_id"] = run.info.run_id
        
        mlflow.pytorch.log_model(
            model, 
            'model',
            signature=signature,
            input_example=X_test_np[0]
        )
        
        mlflow.log_metric('R2', r2)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('F1', f1)
                
        print('Successfully logged metrics and artifacts')
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    print(f"Saving classification report to {evaluation_path}")

    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    print('Evaluation saved successfully')
    
    # artifact_uri = f"s3://mlops-hot-or-meh/mlflow-artifacts"
    # model_folder_name = 'viralhunt'
    # expr_name = "/Users/husarov.pn@ucu.edu.ua/viralhunt3"


    # mlflow.set_tracking_uri("databricks")

    # mlflow.create_experiment(expr_name, artifact_uri)
    # mlflow.set_experiment(expr_name)

    # with mlflow.start_run(run_name=f'Run BertClassifier') as run:
    #     print(f'Run id: {run.info.run_id}')
    #     mlflow.pytorch.log_model(model, 'model')

    #     mlflow.log_metric('R2', r2)
    #     mlflow.log_metric('MAE', mae)
    #     mlflow.log_metric('MSE',  mse)
        
    #     mlflow.log_artifact('../models/bert_classifier.pth')

    #     print('Successfully logged metrics and uploaded artifacts')
