import re             # !!Refer Python RE docs for more information on how to use it
import nltk

from getVocabList import getVocabList

#.....Processing email.....
def ProcessEmail(email):
    # Stripping email
    email=email.lower()                                     # re.sub is used to REPLACE a matched character
    email=re.sub('(http|https)://[^\s]*','httpaddr',email)  # [\s] matches whitespace// [^\s] matches non-white spaces
    email=re.sub('[^\s]+@[^\s]+','emailaddr',email)   # matches any character with @ in-between
    email=re.sub('[<>?,.:/]+',' ',email)              # matches characters inside []
    email=re.sub('[0-9]+','number',email)             # matches numbers from 0-9
    email=re.sub('[$]+','dollar ',email)              # matches $
    email=re.sub('[\s]+',' ',email)                   # matches whitespaces \n,\t,\.....

    print("Processed e-mail :\n\n",email)
    
    ######################
    # Tokenizing Processed email
    tokens=email.split()             # split email to individual tokens or words
    stemmer=nltk.PorterStemmer()     # defining stemmer for use
    
    ######################
    word_index=[]       # [] defines a list or normal array # {} defines a dictionary or associative array
    vocab_dict=getVocabList()                               # dictionary holds key:value, pairs...refer website

    # Indexing email corrosponding to vocab_dict
    for token in tokens:
        token=stemmer.stem(token)                      # stemming     # use token.strip() to be safe....here the email is already stripped from all possible characters
        
        if token in vocab_dict:                        # indexing email with Vocubulary dictionary
            word_index.append(int(vocab_dict[token]))  # use append to add element to empty list/array
                                                       # using int is VERY IMPORTANT. else it will store as characters eg. '86','916'....
    return word_index,vocab_dict