from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

def generate_vocab(dir, min_count, max_files=100):
    dir = os.getcwd() + '/' + dir #Set directory with data
    vocab = []
    text = '' #Need to initialize variables
    
    if max_files == -1: 
        max_files = 10000000 #THERE IS A MORE ELEGANT WAY TO DO THIS BUT I'LL BE DAMNED IF I KNOW IT
   
    i = 0
    
    for file in os.listdir(dir + r'/pos'): # iterate through pos, then neg examples "max_files" times
        text += ' ' + open(dir + r'/pos/' + file, 'r').read() #prevent last word in file "a" and first word in file "b" from mistakenly being connected
        i = i + 1
        if i == max_files:
            break
    
    i = 0
    
    for file in os.listdir(dir + r'/neg'):
        text += ' ' + open(dir + r'/neg/' + file, 'r').read()
        i = i + 1
        if i == max_files:
            break
        
    text = [text]
    v = CountVectorizer(text) 
    v.fit(text)
    rawVocab = v.get_feature_names() #Get vocab words
    vocabArray = v.transform(text).toarray() #Get freq. for each word
        
    for i in range(len(vocabArray[0])): #Across these frequencies,
            if vocabArray[[0], [i]] >= min_count: #Check if they surpass our min_count
                vocab.insert(i, rawVocab[i]) #If so, add to vocab
            else:
                continue
 
    return vocab

def create_word_vector(fname, vocab):
    text = open(fname, 'r', encoding='utf-8').read()
    text = [text]
    v = CountVectorizer(vocabulary = vocab)
    v = v.fit_transform(text)
    feature_vector = v.toarray()
    
    return feature_vector

def load_data(dir, vocab, max_files = 100):
    X, Y = [], []
    i = 0

    if max_files == -1: 
        max_files = 10000000 #BAD CODING PART 2: ELECTRIC BOOGALOO
    
    for file in os.listdir(dir + r'/pos'): #for files in the positive directory
        file = dir + r'/pos/' + file #Set the path for a file
        X.append(create_word_vector(file, vocab))
        Y.append(1)
        
        if i == max_files:
            break
        elif i != max_files:
            i = i + 1
    
    i = 0
    
    for file in os.listdir(dir + r'/neg'):
        file = dir + r'/neg/' + file
        X.append(create_word_vector(file, vocab))
        Y.append(-1)
        
        if i == max_files:
            break
        elif i != max_files:
            i = i + 1
    
    X, X = np.array(X), np.squeeze(X)
    Y = np.array(Y)
    
    return X, Y
