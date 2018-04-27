import pandas as pd
import numpy as np
import nltk
import gensim
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def combine_columns(df,cols):
    all_docs = []
    for index, row in df.iterrows():
        tempstr = ''
        for col in cols:
            try:
                tempstr += str(row[col]) + ' '
            except:
                print('Column Not Found : ', col)
        all_docs.append(tempstr)
    return all_docs

def remove_punctutation(all_docs):
    no_punctuation_docs = []
    for doc in all_docs:
        doc = doc.replace("-", " ")
        no_punctuation_docs.append(re.sub(r'[^\w\s]','',doc))
    return no_punctuation_docs

def tokenizer(all_docs):
    tokenized_docs = []
    for doc in all_docs:
        tokenized_docs.append(word_tokenize(doc))
    return tokenized_docs

def lemmatizer(all_docs):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_docs = []
    for doc in all_docs:
        temp = []
        for token in doc:
            if token.isalpha():
                temp.append(wordnet_lemmatizer.lemmatize(token,"v"))
        lemmatized_docs.append(temp)
    return lemmatized_docs

def stemmer(all_docs):
    porter_stemmer = PorterStemmer()
    stemmed_docs = []
    for doc in all_docs:
        temp = []
        for token in doc:
            temp.append(porter_stemmer.stem(token))
        stemmed_docs.append(temp)
    return stemmed_docs

def untokenizer(all_docs):
    untokenized_docs = []
    for doc in all_docs:
        untokenized_docs.append(" ".join(doc))
    return untokenized_docs

def generate_sentence_vector(tokens, model, vectorizer, tfidf_dense):
    vector = np.zeros(model.vector_size)
    for token in tokens:
        if token in model.wv.vocab and token in vectorizer.vocabulary_:
            vector = vector + model.wv[token] * tfidf_dense[0,vectorizer.vocabulary_[token]]
    return vector

def trainModels(excel_data='hvac_all_issues.xlsx', text_column_list=['Action.Requested', 'General.Comments', 'Task.Comments']):
    df = pd.read_excel(excel_data)
    issue_text = combine_columns(df, text_column_list)

    #REMOVE PUNCTUATION
    punctuation_issues = remove_punctutation(issue_text)

    #Tokenize each issue text
    tokenized_issues = tokenizer(punctuation_issues)

    #lemmatize the tokens
    lemmatized_issues = lemmatizer(tokenized_issues)

    #Stem the words
    stemmed_issues = stemmer(lemmatized_issues)

    #Untokenize the tokens to form sentence again
    untokenized_issues = untokenizer(stemmed_issues)

    stop_words = stopwords.words('english')
    stop_words.append('nan')

    vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, ngram_range=(1,1))
    # tokenize and build vocab
    tfidf_matrix = vectorizer.fit_transform(untokenized_issues)
    tfidf_dense = tfidf_matrix.todense()
    tfidf_file = open("tfidf_vectorizer", "wb")
    pickle.dump(vectorizer, tfidf_file)

    # Train Word2Vec Model from all the tokenized documents after stemming
    model = gensim.models.Word2Vec(stemmed_issues, min_count=1, size=100)
    model.save('word2vec_model')

    issue_features = []
    for i in range(0, len(stemmed_issues)):
        issue_features.append(generate_sentence_vector(stemmed_issues[i], model, vectorizer, tfidf_dense[i]))

    issue_vector_df = pd.DataFrame()
    issue_vector_df['Work Order'] = df['Work.Order']
    issue_vector_df['Action Requested'] = df['Action.Requested']
    issue_vector_df['General Comments'] = df['General.Comments']
    issue_vector_df['Task Comments'] = df['Task.Comments']
    issue_vector_df['Building'] = df['Building']
    issue_vector_df['Campus'] = df['Campus']
    issue_vector_df['Total'] = df['Total']
    issue_vector_df['TextVector'] = issue_features
    #issue_vector_df['DaysToCompletion'] = df['Days.to.Completion']
    issue_vector_df.to_pickle('issue_text_vector.pkl')
    trainX = issue_vector_df['TextVector'].values.tolist()

    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(trainX)
    knn = open('knn', 'wb')
    pickle.dump(neigh, knn)

def getIssueFeatures(issue_string):
    new_issue = [issue_string]
    punctuation_issue = remove_punctutation(new_issue)
    tokenized_issue = tokenizer(punctuation_issue)
    lemmatized_issue = lemmatizer(tokenized_issue)
    stemmed_issue = stemmer(lemmatized_issue)
    untokenized_issue = untokenizer(stemmed_issue)
    with open("tfidf_vectorizer", "rb") as tf:
        vectorizer = pickle.load(tf)
    model = gensim.models.Word2Vec.load('word2vec_model')
    tfidf_matrix = vectorizer.transform(untokenized_issue)
    tfidf_dense = tfidf_matrix.todense()
    return generate_sentence_vector(stemmed_issue[0], model, vectorizer, tfidf_dense[0])

def getSimilarIssues(issue_vector):
    try:
        with open('knn', 'rb') as kfile:
            neigh = pickle.load(kfile)
    except:
        print('Error loading knn model')

    neighbors = neigh.kneighbors([issue_vector], return_distance=False)
    issues_df = pd.read_pickle('issue_text_vector.pkl')
    print(neighbors)
    print(np.shape(neighbors))
    neighbor_df = issues_df.iloc[neighbors[0]]
    neighbor_df = neighbor_df[['Work Order', 'Action Requested', 'Building', 'Total']].copy()
    return neighbor_df

# trainModels()


