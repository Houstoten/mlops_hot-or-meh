import subprocess
import sys

import pandas as pd
import logging
import os
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nltk
import re
import string
import subprocess
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    nltk.data.path.append('/opt/ml/processing/utils')
    df = pd.read_csv('/opt/ml/processing/input/ProductHuntProducts.csv')

    df.drop_duplicates(subset='id', keep="first", inplace=True)

    df['target'] = df['votesCount'] + df['commentsCount']

    scaler = MinMaxScaler()
    df['target'] = scaler.fit_transform(np.maximum(0, np.log(df[['target']])))

    df['target_binary'] = (df['target'] >= df['target'].mean()).astype(int)

    df['description'] = df['description'].fillna('Empty').astype(str).str.lower()
    df['tagline'] = df['tagline'].fillna('Empty').astype(str).str.lower()

    def clean_string(text, stem="None"):

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

    df['text'] = df['tagline'] + " " + df['description']
    df['text'] = df['text'].apply(lambda text: clean_string(text, stem='Lem'))

    train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].to_numpy(), df['target_binary'].to_numpy(), test_size=0.1, random_state=42)

    train_features_path = os.path.join("/opt/ml/processing/split/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/split/train", "train_labels.csv")

    test_features_path = os.path.join("/opt/ml/processing/split/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/split/test", "test_labels.csv")

    print(f"Saving training data to {train_features_path}")
    pd.DataFrame(train_texts).to_csv(train_features_path, index=False)
    # train_texts.to_csv(train_features_path, index=False, header=False)

    print(f"Saving test data to {test_features_path}")
    pd.DataFrame(test_texts).to_csv(test_features_path, index=False)

    print(f"Saving training labels to {train_labels_output_path}")
    pd.DataFrame(train_labels).to_csv(train_labels_output_path, index=False)

    print(f"Saving test labels to {test_labels_output_path}")
    pd.DataFrame(test_labels).to_csv(test_labels_output_path, index=False)
# print("Hi there out:)")

# if __name__ == "__main__":
#     print("Hi there:)")

# import logging
# import time

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def main():
#     logger.info("Starting the script.")
    
#     # Simulate some processing
#     try:
#         for i in range(5):
#             logger.info(f"Processing step {i+1}")
#             time.sleep(1)  # Simulate processing time
#         logger.info("Processing completed successfully.")
#     except Exception as e:
#         logger.error("An error occurred during processing.", exc_info=True)

# if __name__ == "__main__":
#     logger.info("Hi there:)")
#     main()
