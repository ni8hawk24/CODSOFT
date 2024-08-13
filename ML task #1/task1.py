import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text


def parse_dataset(file_path, has_genre=False):
    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if has_genre:
                if len(parts) != 4:
                    
                    continue
                serial, title, genre, description = parts
                parsed_data.append([serial, title, genre, description])
            else:
                if len(parts) != 3:
                    
                    continue
                serial, title, description = parts
                parsed_data.append([serial, title, description])


    if has_genre:
        df = pd.DataFrame(parsed_data, columns=['Serial', 'Title', 'Genre', 'Description'])
    else:
        df = pd.DataFrame(parsed_data, columns=['Serial', 'Title', 'Description'])

  
    df['Description'] = df['Description'].astype(str).apply(preprocess_text)
    return df


train_data_path = 'train_data.txt'
test_data_path = 'test_data_solution.txt'


train_df = parse_dataset(train_data_path, has_genre=True)
test_df = parse_dataset(test_data_path, has_genre=False)


X_train = train_df['Description']
y_train = train_df['Genre']


pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipeline.fit(X_train, y_train)


joblib.dump(pipeline, 'model.pkl')


model = joblib.load('model.pkl')

def predict_genre(description):
    
    processed_text = preprocess_text(description)
    
   
    prediction = model.predict([processed_text])
    
    return prediction[0]


user_description = input("Enter a movie description to predict its genre: ")


predicted_genre = predict_genre(user_description)
print(f'Predicted genre: {predicted_genre}')
