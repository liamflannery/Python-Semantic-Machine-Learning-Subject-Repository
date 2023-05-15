import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('gutenberg')
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pandas as pd


# Task 1 (1 mark)
import collections
nltk.download('universal_tagset')
def count_pos(document, pos):
    """Return the number of occurrences of words with a given part of speech. To find the part of speech, use 
    NLTK's "Universal" tag set. To find the words of the document, use NLTK's sent_tokenize and word_tokenize.
    >>> count_pos('austen-emma.txt', 'NOUN')
    31998
    >>> count_pos('austen-sense.txt', 'VERB')
    25074
    """
    word_tag = []
    for sentance in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document)):
        word = nltk.word_tokenize(sentance)
        word_tag += nltk.pos_tag(word, tagset='universal')

    
    tag = [w for [n,w] in word_tag]
    counter = collections.Counter(tag)
    return counter[pos]

# Task 2 (2 marks)
def get_top_stem_bigrams(document, n):
    """Return the n most frequent bigrams of stems. Return the list sorted in descending order of frequency.
    The stems of words in different sentences cannot form a bigram. To stem a word, use NLTK's Porter stemmer.
    To find the words of the document, use NLTK's sent_tokenize and word_tokenize.
    >>> get_top_stem_bigrams('austen-emma.txt', 3)
    [(',', 'and'), ('.', "''"), (';', 'and')]
    >>> get_top_stem_bigrams('austen-sense.txt',4)
    [(',', 'and'), ('.', "''"), (';', 'and'), (',', "''")]
    """
    stemmer = nltk.PorterStemmer()
    words = []
    bigrams = list()
    for sentance in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document)):
        words = [stemmer.stem(word) for word in nltk.word_tokenize(sentance)]
        bigrams += list(nltk.bigrams(words))
    
    counter = collections.Counter(bigrams).most_common(n)
    counter_small = [x for [x,y] in counter]
    return counter_small


# Task 3 (2 marks)
def get_same_stem(document, word):
    """Return the list of words that have the same stem as the word given, and their frequencies. 
    To find the stem, use NLTK's Porter stemmer. To find the words of the document, use NLTK's 
    sent_tokenize and word_tokenize. The resulting list must be sorted alphabetically.
    >>> get_same_stem('austen-emma.txt','respect')[:5]
    [('Respect', 2), ('respect', 41), ('respectability', 1), ('respectable', 20), ('respectably', 1)]
    >>> get_same_stem('austen-sense.txt','respect')[:5]
    [('respect', 22), ('respectability', 1), ('respectable', 14), ('respectably', 1), ('respected', 3)]
    """
    stemmer = nltk.PorterStemmer()
    words = dict()
    targetStem = stemmer.stem(word)
    for sentance in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document)):
        for currentWord in nltk.word_tokenize(sentance):
            if stemmer.stem(currentWord) == targetStem:
                if(currentWord in words.keys()):
                    words[currentWord] += 1
                else:
                    words[currentWord] = 1
    listWords = words.items()
    sortedWords = sorted(listWords)
    return sortedWords

# Task 4 (2 marks)
def most_frequent_after_pos(document, pos):
    """Return the most frequent word after a given part of speech, and its frequency. Do not consider words
    that occur in the next sentence after the given part of speech.
    To find the part of speech, use NLTK's "Universal" tagset.
    >>> most_frequent_after_pos('austen-emma.txt','VERB')
    [('not', 1932)]
    >>> most_frequent_after_pos('austen-sense.txt','NOUN')
    [(',', 5310)]
    """
    followWords = dict()
    prevWord = False;
    for sentance in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document)):
         currentWord = nltk.word_tokenize(sentance)
         for tokenWord in nltk.pos_tag(currentWord, tagset='universal'):
             currentPart = tokenWord[1]
             
             if(prevWord):
                 if(tokenWord[0] in followWords.keys()):
                    followWords[tokenWord[0]] += 1
                 else:
                    followWords[tokenWord[0]] = 1
             prevWord = False;
             if(currentPart == pos):
                 prevWord = True;         
              
    followWords = sorted(followWords.items(), key = lambda item: item[1], reverse = True)
    return [followWords[0]]

# Task 5 (2 marks)
def get_word_tfidf(text):
    """Return the tf.idf of the words given in the text. If a word does not have tf.idf information or is zero, 
    then do not return its tf.idf. The reference for computing tf.idf is the list of documents from the NLTK 
    Gutenberg corpus. To compute the tf.idf, use sklearn's TfidfVectorizer with the option to remove the English 
    stop words (stop_words='english'). The result must be a list of words sorted in alphabetical order, together 
    with their tf.idf.
    >>> get_word_tfidf('Emma is a respectable person')
    [('emma', 0.8310852062844262), ('person', 0.3245184217533661), ('respectable', 0.4516471784898886)]
    >>> get_word_tfidf('Brutus is a honourable person')
    [('brutus', 0.8405129362379974), ('honourable', 0.4310718596448824), ('person', 0.32819971943754456)]
    """
    tfidfVectorizer = TfidfVectorizer(input='content',stop_words='english')
    data = [nltk.corpus.gutenberg.raw(f) for f in nltk.corpus.gutenberg.fileids()]
    tfidf = tfidfVectorizer.fit_transform(data)
    df = pd.DataFrame(tfidf[0].T.todense(), index=tfidfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    result = []
    df = df["TF-IDF"].to_dict()
    dfItems = df.items()
   
    for word in text.split(" "):
         for i in dfItems:
            if(i[0] == word.lower() and i[1] > 0):
                result += i
    return result


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
