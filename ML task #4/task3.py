import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report




data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Rename columns for clarity
data.columns = ['label', 'text', 'v3', 'v4', 'v5']
data = data[['label', 'text']]


def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

data['text'] = data['text'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)


tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


def predict_message(message):

    message = clean_text(message)
    message_tfidf = tfidf.transform([message])
    prediction = model.predict(message_tfidf)
    return prediction[0]


while True:
    user_message = input("Enter a message to check if it's spam or not (type 'exit' to quit): ")
    if user_message.lower() == 'exit':
        break
    prediction = predict_message(user_message)
    print(f"The message is predicted to be: {prediction}")
