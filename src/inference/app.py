from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import logging
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
import nltk
import re
import string
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import io
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# nltk.download('stopwords', download_dir='./utils/')
# nltk.download('wordnet', download_dir='./utils/')
# command = "unzip ./utils/corpora/wordnet.zip -d ./utils/corpora -y"

# subprocess.run(command.split())

nltk.data.path.append('./utils')

app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)



def process_text(tagline, description, stem=None):

    text = tagline.lower() + " " + description.lower() 

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    # Note: that this line can be augmented and used over
    # to replace any characters with nothing or a space
    text = re.sub(r'\n', '', text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string

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

    model = BERTClassifier('bert-base-uncased', 2)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(load_object_from_s3('bert_classifier.pth'), map_location=torch.device('cpu')))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer

model, tokenizer = load_model_for_inference()


transport = RequestsHTTPTransport(
    url='https://api.producthunt.com/v2/api/graphql',
    headers={'Authorization': f'Bearer {os.getenv("PH_ACCESS_TOKEN")}'}
)

client = Client(transport=transport, fetch_schema_from_transport=True)

query = gql("""
query fetchPost($slug: String){
  post(slug:$slug) {
    tagline,
    description,
  }
}
""")

def predict_hotness(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return "Hot" if preds.item() == 1 else "Not Hot"


@app.route('/predict', methods=['POST'])
def predict():
    result = client.execute(query, variable_values={"slug": request.json['url'].split('/')[-1]});
    # print(process_text(result['post']['tagline'], result['post']['description'], stem='Lem'))

    return jsonify({'prediction': predict_hotness(process_text(result['post']['tagline'], result['post']['description'], stem='Lem'), model, tokenizer, device='cpu')}), 200
    # try:
    #     data = request.json['short_description']
    #     app.logger.info(f"Received data: {data}")
    #     if not data:
    #         return jsonify({'error': 'No text provided'}), 400
    #     vect_data = vectorizer.transform([data])
    #     prediction = model.predict(vect_data)
    #     app.logger.info(f"Prediction: {prediction[0]}")
    #     return jsonify({'prediction': prediction[0]}), 200
    # except Exception as e:
    #     app.logger.error(f"Error in prediction: {e}")
    #     return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)