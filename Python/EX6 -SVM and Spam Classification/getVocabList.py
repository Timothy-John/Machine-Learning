
#.....Converting vocab list to dictionary.....
def getVocabList():
    # METHOD 1 :    uses true intution of {} or dict. ie, key-value pair.....refer website 
    vocab=open('vocab.txt')   # open considers words but if you read the opened file it will consider characters which is not desirable in dictionary

    vocab_dict={}       # {} defines an empty dict or associative array
    for rows in vocab:
        key,value =rows.split()      # split by space. open vocab.txt to get better idea
        vocab_dict[value]=key

    # METHOD 2 :     uses csv to split key and value same as above
    """
    import csv
    vocab_dict={}
    vocab=open('vocab.txt')
    vocab=csv.reader(vocab,delimiter='\t')
    for rows in vocab:
        vocab_dict[rows[1]]=rows[0]
    """
    return vocab_dict