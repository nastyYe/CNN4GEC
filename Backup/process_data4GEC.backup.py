import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import configure
import pickle

def build_data(data_folder,clean_string=True):
    """
    Loads data and split into 10 folds.
    vocab: dict key:word value: the number of word occurence in the corpus
    """

    train_file = data_folder[0]
    test_file = data_folder[1]

    vocab = defaultdict(float)

    revs_train = readTrainTest(train_file,vocab)
    revs_test  = readTrainTest(test_file,vocab)

    return revs_train,revs_test,vocab
    

def readTrainTest(input1,vocab,clean_string=True):
    revs = []
    begin = 7
    with open(input1,'r') as fr:
        while True:
            line = fr.readline()[:-1]
            if not line:
                break
            revs.append(line)

            if clean_string:
                orig_rev = clean_str(line.strip().lower())

            else:
                orig_rev = line.strip().lower()

            words = set(orig_rev.split()[begin:])
            for word in words:
                vocab[word] += 1

    return revs

def get_W(word_vecs, k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    word_idx_map : word's id in W
    word_vecs: dict key=word val=vec
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


#load from the word2vec.txt  or google-news-corpus
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

#load the word2vec's vector
def load_word2vec(fname):
    word_vecs = pickle.load(open(fname,'rb'))

    for word in word_vecs:
        word_vecs[word] = np.array(word_vecs[word],dtype='float32')

    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df, k):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def backup():
    #w2v_file = sys.argv[1]    # word2vec file to parse the vector! 
    w2v_file = configure.fTrainTestVecArt
    vectorL = 50 

    #data_folder = ["rt-polarity.pos","rt-polarity.neg"]   # train and test data 
    data_folder = [configure.fTrainTokenArt,configure.fTestTokenArt]
    
    print "loading data...",  

    #revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)  #revs: the list of Datatum   vocab: dict of word-frequency
    revs_train,revs_test,vocab = build_data(data_folder, clean_string=True)  #revs: the list of Datatum   vocab: dict of word-frequency

    #max_l = np.max(pd.DataFrame(revs)["num_words"])  #record the max length of the sentence in the corpus!
    max_l = 8    # Set the max length of the sentence 8

    print "data loaded!"
    print "number of sentences: " + str(len(revs_train)+len(revs_test))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",

    """read the vector in the vocab word, if we don't use the google-news-corpus , we can use other method to replace!"""
    #w2v = load_bin_vec(w2v_file, vocab) 
    w2v = load_word2vec(w2v_file)

    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #parameter : (word_vecs, vocab, min_df=1, k=300)
    add_unknown_words(w2v, vocab,1,vectorL)
    #get_W(word_vecs, k=300)
    W, word_idx_map = get_W(w2v,vectorL)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab,1,vectorL)  #rand initial the word's vector 
    W2, _ = get_W(rand_vecs,vectorL)  # because the word's vector is initialed so the word and it's vector donot need to match
    cPickle.dump([revs_train,revs_test, W, W2, word_idx_map, vocab], open("tmp/mr.p", "wb"))
    print "dataset created!"

def processArtOrDet():

    #w2v_file = sys.argv[1]    # word2vec file to parse the vector! 
    w2v_file = configure.fTrainTestVecArt
    vectorL = 50 

    #data_folder = ["rt-polarity.pos","rt-polarity.neg"]   # train and test data 
    data_folder = [configure.fTrainTokenArt,configure.fTestTokenArt]
    
    print "loading data...",  

    #revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)  #revs: the list of Datatum   vocab: dict of word-frequency
    revs_train,revs_test,vocab = build_data(data_folder, clean_string=True)  #revs: the list of Datatum   vocab: dict of word-frequency

    #max_l = np.max(pd.DataFrame(revs)["num_words"])  #record the max length of the sentence in the corpus!
    max_l = 8    # Set the max length of the sentence 8

    print "data loaded!"
    print "number of sentences: " + str(len(revs_train)+len(revs_test))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",

    """read the vector in the vocab word, if we don't use the google-news-corpus , we can use other method to replace!"""
    #w2v = load_bin_vec(w2v_file, vocab) 
    w2v = load_word2vec(w2v_file)

    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #parameter : (word_vecs, vocab, min_df=1, k=300)
    add_unknown_words(w2v, vocab,1,vectorL)
    #get_W(word_vecs, k=300)
    W, word_idx_map = get_W(w2v,vectorL)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab,1,vectorL)  #rand initial the word's vector 
    W2, _ = get_W(rand_vecs,vectorL)  # because the word's vector is initialed so the word and it's vector donot need to match
    cPickle.dump([revs_train,revs_test, W, W2, word_idx_map, vocab], open("tmp/mr.p", "wb"))
    print "dataset created!"
    

if __name__=="__main__":    
    #w2v_file = sys.argv[1]    # word2vec file to parse the vector! 
    w2v_file = configure.fTrainTestVecArt
    vectorL = 50 

    #data_folder = ["rt-polarity.pos","rt-polarity.neg"]   # train and test data 
    data_folder = [configure.fTrainTokenArt,configure.fTestTokenArt]
    
    print "loading data...",  

    #revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)  #revs: the list of Datatum   vocab: dict of word-frequency
    revs_train,revs_test,vocab = build_data(data_folder, clean_string=True)  #revs: the list of Datatum   vocab: dict of word-frequency

    #max_l = np.max(pd.DataFrame(revs)["num_words"])  #record the max length of the sentence in the corpus!
    max_l = 8    # Set the max length of the sentence 8

    print "data loaded!"
    print "number of sentences: " + str(len(revs_train)+len(revs_test))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",

    """read the vector in the vocab word, if we don't use the google-news-corpus , we can use other method to replace!"""
    #w2v = load_bin_vec(w2v_file, vocab) 
    w2v = load_word2vec(w2v_file)

    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #parameter : (word_vecs, vocab, min_df=1, k=300)
    add_unknown_words(w2v, vocab,1,vectorL)
    #get_W(word_vecs, k=300)
    W, word_idx_map = get_W(w2v,vectorL)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab,1,vectorL)  #rand initial the word's vector 
    W2, _ = get_W(rand_vecs,vectorL)  # because the word's vector is initialed so the word and it's vector donot need to match
    cPickle.dump([revs_train,revs_test, W, W2, word_idx_map, vocab], open("tmp/mr.p", "wb"))
    print "dataset created!"
