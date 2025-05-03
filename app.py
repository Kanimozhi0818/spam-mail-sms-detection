import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the real SMS Spam Collection dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert label to binary: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply preprocessing
df['text'] = df['text'].apply(preprocess)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("\n🔍 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# User input loop
print("\n📬 Spam Detector Ready. Type 'exit' to quit.")
while True:
    user_input = input("\n📧 Enter email content: ")
    if user_input.lower() == 'exit':
        print("Exiting Spam Detector.")
        break

    user_input_clean = preprocess(user_input)
    user_input_vec = vectorizer.transform([user_input_clean])
    prediction = model.predict(user_input_vec)

    print("⚠️ SPAM" if prediction[0] == 1 else "✅ NOT SPAM")
