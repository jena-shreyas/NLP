import numpy as np
import random
import nltk
from pos_tagger import POSTagger
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

nltk.download('movie_reviews')

def tag_movie_reviews(tagger: POSTagger, reviews: list[list]):
    tagged_reviews = []

    print("Tagging movie reviews...")
    for review in tqdm(reviews):
        tagged_review = []
        for sentence in review:
            tagged_review.append(tagger.viterbi(sentence))
        tagged_reviews.append(tagged_review)

    return tagged_reviews


def process_tagged_reviews(tagged_reviews: list[list[list]]):
    processed_reviews = []
    # all tags for adjective, noun, verb, adverb
    # keep only words with these tags
    tags_keep = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS'}

    print("Processing tagged reviews...")
    for review in tqdm(tagged_reviews):
        processed_review = []
        for sentence in review:
            processed_sentence = []
            for word, tag in sentence:
                if tag in tags_keep:
                    processed_sentence.append(word)
            processed_review += processed_sentence

        processed_reviews.append(processed_review)

    return processed_reviews

# Train the POS tagger
pos_tagger = POSTagger()
pos_tagger.train()

documents = [(list(movie_reviews.sents(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Split the data into train and test sets
train_size = int(0.8 * len(documents))

train_reviews = [review for review, _ in documents[:train_size]]
test_reviews = [review for review, _ in documents[train_size:]]

# function to concatenate all sentences in a review (this output will be used to train the Word2Vec model)
def concat_review_sents(reviews):
    return [[word for sent in review for word in sent] for review in reviews]

concat_train_reviews = concat_review_sents(train_reviews)

train_labels = [label for _, label in documents[:train_size]]
test_labels = [label for _, label in documents[train_size:]]

print("Tagging reviews...")
tagged_train_reviews = tag_movie_reviews(pos_tagger, train_reviews)
tagged_test_reviews = tag_movie_reviews(pos_tagger, test_reviews)

print("Processing tagged reviews...")
processed_train_reviews = process_tagged_reviews(tagged_train_reviews)
processed_test_reviews = process_tagged_reviews(tagged_test_reviews)

# train Word2vec model on train reviews
model = Word2Vec(concat_train_reviews, vector_size=512, alpha=0.05, window=5, min_count=1)

train_features = list()
# Vectorize the text data using the trained Word2Vec model
for review in processed_train_reviews:
    review_emb = np.mean([model.wv[word] for word in review], axis=0)
    train_features.append(review_emb)

# define SVM model and train it on the training vectors
clf = SVC(kernel='linear')
clf.fit(train_features, train_labels)

# Vectorize the test set
test_features = list()
for review in processed_test_reviews:
    word_embs = list()
    for word in review:
        if word in model.wv:
            word_embs.append(model.wv[word])
    review_emb = np.mean(word_embs, axis=0)
    test_features.append(review_emb)

# Test the classifier on the test set
test_predictions = clf.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
classification_report_str = classification_report(test_labels, test_predictions)
print("\nTest Accuracy:", test_accuracy)
print("Test classification Report:\n", classification_report_str)

# save outputs to file
with open('improved_output.txt', 'w') as f:
    print("\nTest Accuracy:", test_accuracy, file=f)
    print("Test classification Report:\n", classification_report_str, file=f)
