import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load the DataFrame from the CSV file
df = pd.read_csv('twitter_data.csv')

# Initialize the sentiment analysis model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Define labels for sentiment categories
labels = ['Negative', 'Neutral', 'Positive']

# Function to preprocess tweets
def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    return " ".join(tweet_words)

# Set maximum sequence length
max_length = 128

# Perform sentiment analysis on each tweet in the DataFrame
sentiments = []

for index, tweet in enumerate(df['Tweet']):
    preprocessed_tweet = preprocess_tweet(tweet)
    encoded_tweet = tokenizer.encode_plus(
        preprocessed_tweet,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = model(**encoded_tweet)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    sentiment = {label: score.item() for label, score in zip(labels, scores)}
    sentiments.append(sentiment)
    print(f'Progress: {index+1}/{len(df["Tweet"])}')

# Append the sentiment scores to the DataFrame
sentiment_df = pd.DataFrame(sentiments)
df2 = pd.concat([df, sentiment_df], axis=1)

# Save the modified DataFrame
df2.to_csv("Twitter_Data.csv", index=False)
