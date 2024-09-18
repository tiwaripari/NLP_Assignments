import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Define the paths to the dataset files
positive_file = 'rt-polaritydata\\rt-polarity.pos' #Change the address accordingly
negative_file = 'rt-polaritydata\\rt-polarity.neg' #Change the address accordingly

# Load the data with fallback encoding to handle potential issues
def load_data(pos_file, neg_file):
    with open(pos_file, 'r', encoding='ISO-8859-1') as file:
        positive_texts = file.readlines()
    with open(neg_file, 'r', encoding='ISO-8859-1') as file:
        negative_texts = file.readlines()
    return positive_texts, negative_texts


positive_texts, negative_texts = load_data(positive_file, negative_file)
print(f"Loaded {len(positive_texts)} positive and {len(negative_texts)} negative texts.")

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine (SVM)": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Function to prioritize adjectives while retaining full context
def get_adjective_weighted_text(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    weighted_words = []

    for word, pos in pos_tags:
        weighted_words.append(word)  # Keep all words to retain the full context
        if pos.startswith('JJ'):  # Repeat adjectives to give them more weight
            weighted_words.append(word)

    return ' '.join(weighted_words)

# Preprocess the positive and negative texts
positive_weighted_texts = [get_adjective_weighted_text(text) for text in positive_texts]
negative_weighted_texts = [get_adjective_weighted_text(text) for text in negative_texts]

# Split data into train, validation, and test sets as per your specification
# Training set: First 4000 positive and 4000 negative
train_texts = positive_weighted_texts[:4000] + negative_weighted_texts[:4000]
train_labels = [1] * 4000 + [0] * 4000

# Validation set: Next 500 positive and 500 negative
val_texts = positive_weighted_texts[4000:4500] + negative_weighted_texts[4000:4500]
val_labels = [1] * 500 + [0] * 500

# Test set: Final 831 positive and 831 negative
test_texts = positive_weighted_texts[4500:5331] + negative_weighted_texts[4500:5331]
test_labels = [1] * 831 + [0] * 831

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

X_train_dense = X_train.todense()

#check whether matrix contains non_zero terms
row_sums = np.sum(X_train_dense, axis=1)
non_zero_rows = np.sum(row_sums != 0)
row_sums

X_train

for clf_name, clf in classifiers.items():
    print(f"Training {clf_name}...")

    # Training the classifier
    clf.fit(X_train, train_labels)

    # Predicting on validation set
    y_val_pred = clf.predict(X_val)

    # Evaluate the classifier
    print(f"Validation results for {clf_name}:")
    print(f"Accuracy: {accuracy_score(val_labels, y_val_pred):.4f}")
    print("Classification Report:")
    print(classification_report(val_labels, y_val_pred))
    print("-" * 80)

best_clf = classifiers["Naive Bayes"]  # For example, let's say LR performed best
y_test_pred = best_clf.predict(X_test)

print("Final Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(test_labels, y_test_pred):.4f}")
print("Classification Report:")
print(classification_report(test_labels, y_test_pred))

# Calculate confusion matrix for TP, TN, FP, FN
conf_matrix = confusion_matrix(test_labels, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract TP, TN, FP, FN from the confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")