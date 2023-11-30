import numpy as np
import nltk
import random
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Download the movie_reviews corpus
nltk.download('movie_reviews')

# Load the movie_reviews corpus and extract reviews and labels
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Split the data into train, validation, and test sets
train_size = int(0.8 * len(documents))

train_tok_sentences = [words for words, _ in documents[:train_size]]
test_tok_sentences = [words for words, _ in documents[train_size:]]

# print(train_tok_sentences[0])
train_labels = [label for _, label in documents[:train_size]]
test_labels = [label for _, label in documents[train_size:]]

# Train a Word2Vec embedding model on the training set
model = Word2Vec(train_tok_sentences, vector_size=512, alpha=0.05, window=5, min_count=1)

train_features = list()
# Vectorize the text data using the trained Word2Vec model
for sentence in train_tok_sentences:
    sent_emb = np.mean([model.wv[word] for word in sentence], axis=0)
    train_features.append(sent_emb)

# define SVM model and train it on the training vectors
clf = SVC(kernel='linear')
clf.fit(train_features, train_labels)

# Vectorize the test set
test_features = list()
for sentence in test_tok_sentences:
    word_embs = list()
    for word in sentence:
        if word in model.wv:
            word_embs.append(model.wv[word])
    sent_emb = np.mean(word_embs, axis=0)
    test_features.append(sent_emb)

# Test the classifier on the test set
test_predictions = clf.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
classification_report_str = classification_report(test_labels, test_predictions)
print("\nTest Accuracy:", test_accuracy)
print("Test classification Report:\n", classification_report_str)

with open('vanilla_output.txt', 'w') as f:
    print("\nTest Accuracy:", test_accuracy, file=f)
    print("Test classification Report:\n", classification_report_str, file=f)
