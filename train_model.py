"""
Script to train and save the SVC model.
This runs during Render build to generate models/svc.pkl
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def train_and_save_model():
    print("Loading training data...")
    df = pd.read_csv('datasets/Training.csv')
    
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    print("Encoding labels...")
    le = LabelEncoder()
    le.fit(y)
    Y = le.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
    
    print("Training SVC model...")
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    
    print(f"Model accuracy: {svc.score(X_test, y_test)}")
    
    os.makedirs('models', exist_ok=True)
    
    print("Saving model to models/svc.pkl...")
    with open('models/svc.pkl', 'wb') as f:
        pickle.dump(svc, f)
    
    print("Model training complete!")

if __name__ == "__main__":
    train_and_save_model()
