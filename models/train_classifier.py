import nltk
import pandas as pd
import pickle
import re
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine
from workspace_utils import active_session

nltk.download('stopwords')

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('clean_disaster_data', engine)
    engine.dispose()
    
    X = df.message
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize message text. First, detects urls, then replaces those with the string "urlplaceholder". Second, tokenizes then lemmatizes non-stop words. Finally, returns 
    list of resulting lemmatized tokens
    
    Arguments:
    - The message text
    
    Returns:
    - List of resulting lemmatized tokens
    '''
    # regex borrowed from this link:
    # https://www.geeksforgeeks.org/python-check-url-string/
    url_pat = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(url_pat, text)
    urls = [url[0] for url in urls]
    for url in urls:
        text = text.replace(url, 'urlplaceholder')
    
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stop_words]
    
    return lemmatized_tokens


def build_model():
    '''
    Builds ML model pipeline
    
    Returns:
    ML Model Pipeline
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))))
                        ])
    
    parameters = {
    'clf__estimator__learning_rate': [0.1, 0.2],
    'clf__estimator__n_estimators': [50, 100, 200]
    }
    
    pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=2)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    f_score = f1_score(Y_test, Y_pred, average='weighted')
    print('f1-score is: ', f_score)
    
    for i, category in enumerate(category_names):
        print('Category: ', category)
        print(classification_report(Y_test.loc[:, category].values, Y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    with active_session():
        main()