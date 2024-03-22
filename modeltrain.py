import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv('data.csv')

# Fill NaN values in the 'Review text' column with empty strings
data['Review text'].fillna('', inplace=True)

# Basic preprocessing
# Let's assume ratings > 3 are positive (1) and <= 3 are negative (0)
data['Sentiment'] = data['Ratings'].apply(lambda x: 1 if x > 3 else 0)
X = data['Review text']
y = data['Sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')

# Save the model
joblib.dump(model, 'sentiment_model.pkl')