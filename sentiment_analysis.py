import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from afinn import Afinn
import xgboost as xgb  

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Initialize AFINN for sentiment scoring
afinn = Afinn()

# Define the paths to the dataset files (change the paths accordingly)
positive_file = 'rt-polaritydata\\rt-polarity.pos'
negative_file = 'rt-polaritydata\\rt-polarity.neg'

# Load the data with fallback encoding to handle potential issues
def load_data(pos_file, neg_file):
    with open(pos_file, 'r', encoding='ISO-8859-1') as file:
        positive_texts = file.readlines()
    with open(neg_file, 'r', encoding='ISO-8859-1') as file:
        negative_texts = file.readlines()
    return positive_texts, negative_texts

positive_texts, negative_texts = load_data(positive_file, negative_file)
print(f"Loaded {len(positive_texts)} positive and {len(negative_texts)} negative texts.")

# Define classifiers with optimized parameters, including XGBoost
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')  # Added XGBoost
}

# Custom function to apply sentiment weights based on POS and AFINN sentiment scores
def apply_afinn_weights(texts, vectorizer):
    # Transform the text into the TF-IDF matrix
    X = vectorizer.transform(texts).toarray()  # Convert to dense for easier modification
    
    # Get feature names (i.e., the vocabulary learned by the vectorizer)
    feature_names = vectorizer.get_feature_names_out()

    # Tokenize each text, get POS tags, and modify weights based on AFINN sentiment scores
    for i, text in enumerate(texts):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        for word, pos in pos_tags:
            if word in feature_names:
                score = afinn.score(word)
                idx = feature_names.tolist().index(word)
                # Adjust weights based on sentiment score and POS tag
                if pos.startswith('JJ') or pos.startswith('NN'):
                    X[i, idx] *= 1.5 + abs(score)  # Higher weight for adjectives/nouns
                elif pos.startswith('RB'):
                    X[i, idx] *= 1.3 + abs(score)  # Higher weight for adverbs
                else:
                    X[i, idx] *= 1.0 + abs(score)  # Default weight with sentiment
                
    return X

# Split data into train, validation, and test sets
train_texts = positive_texts[:4000] + negative_texts[:4000]
train_labels = [1] * 4000 + [0] * 4000

val_texts = positive_texts[4000:4500] + negative_texts[4000:4500]
val_labels = [1] * 500 + [0] * 500

test_texts = positive_texts[4500:5331] + negative_texts[4500:5331]
test_labels = [1] * 831 + [0] * 831

# Vectorize the text using TF-IDF with n-grams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts).toarray()
X_val = vectorizer.transform(val_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()

# Apply AFINN-based weighting to the TF-IDF matrix
X_train = apply_afinn_weights(train_texts, vectorizer)
X_val = apply_afinn_weights(val_texts, vectorizer)
X_test = apply_afinn_weights(test_texts, vectorizer)

# Training classifiers
for clf_name, clf in classifiers.items():
    print(f"Training {clf_name}...")

    # Train the classifier
    clf.fit(X_train, train_labels)

    # Predicting on validation set
    y_val_pred = clf.predict(X_val)

    # Evaluate the classifier
    print(f"Validation results for {clf_name}:")
    print(f"Accuracy: {accuracy_score(val_labels, y_val_pred):.4f}")
    print("Classification Report:")
    print(classification_report(val_labels, y_val_pred))
    print("-" * 80)

# Voting ensemble classifier with optimized classifiers
ensemble_clf = VotingClassifier(estimators=[
    ('nb', MultinomialNB()),
    ('lr', LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)),
], voting='hard', n_jobs=-1)

ensemble_clf.fit(X_train, train_labels)

# Predicting on validation set
y_val_pred = ensemble_clf.predict(X_val)

# Evaluate the classifier
print(f"Validation results for Ensemble:")
print(f"Accuracy: {accuracy_score(val_labels, y_val_pred):.4f}")
print("Classification Report:")
print(classification_report(val_labels, y_val_pred))
print("-" * 80)

# Final evaluation on test set using Naive Bayes (or replace with best performing classifier)
best_clf = classifiers["Logistic Regression"]
y_test_pred = best_clf.predict(X_test)

print("Final Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(test_labels, y_test_pred):.4f}")
print("Classification Report:")
print(classification_report(test_labels, y_test_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract TP, TN, FP, FN from the confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
