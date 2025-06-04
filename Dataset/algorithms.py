import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv("Dataset/phishingEmail.csv")
    df.isnull().sum()
    df.drop(["Unnamed: 0"], axis=1, inplace=True, errors="ignore")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    print("Dimension of the row data:",df.shape)   
    # Create the bar chart
    fig = px.bar(df['Email Type'].value_counts(), x=df['Email Type'].value_counts().index, y=df['Email Type'].value_counts().values,
             color=['blue', 'red'], labels={'x': 'Category', 'y': 'Count'},
             title="Categorical Distribution")
    fig.show()
 
    # Create the pie chart
    fig_pie = px.pie(df['Email Type'].value_counts(), names=df['Email Type'].value_counts().index,
                 values=df['Email Type'].value_counts().values, title="Categorical Distribution")
    fig_pie.show()

    le = LabelEncoder()
    df["Email Type"] = le.fit_transform(df["Email Type"])

    def preprocess_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove hyperlinks
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    df["Email Text"] = df["Email Text"].apply(preprocess_text)

    # TF-IDF Feature Extraction
    tf = TfidfVectorizer(stop_words="english", max_features=10000)
    feature_x = tf.fit_transform(df["Email Text"]).toarray()
    y_tf = np.array(df['Email Type'])  # Convert labels to numpy array

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(feature_x, y_tf, train_size=0.8, random_state=0)
    return df, x_train, x_test, y_train, y_test

def generate_word_clouds(df):
    # Combine all email texts into one string
    all_mails = " ".join(df['Email Text'])
    
    # Word Cloud with stopwords removed
    word_cloud1 = WordCloud(stopwords="english", width=800, height=400, background_color='white').generate(all_mails)
    plt.figure(figsize=(10, 6))
    plt.imshow(word_cloud1, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Stopwords")
    plt.show()
    
    # Word Cloud with up to 10,000 unique words
    word_cloud2 = WordCloud(width=800, height=400, background_color='white', max_words=10000).generate(all_mails)
    plt.figure(figsize=(10, 6))
    plt.imshow(word_cloud2, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Unique Words")
    plt.show()

def load_and_preprocess_kfold():
    df = pd.read_csv("Dataset/phishingEmail.csv")
    df.drop(["Unnamed: 0"], axis=1, inplace=True, errors="ignore")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    le = LabelEncoder()
    df["Email Type"] = le.fit_transform(df["Email Type"])

    def preprocess_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove hyperlinks
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    df["processed_text"] = df["Email Text"].apply(preprocess_text)

    tf = TfidfVectorizer(stop_words="english", max_features=10000)
    X = tf.fit_transform(df["processed_text"]) 
    y = np.array(df['Email Type'])

    return df, X, y

