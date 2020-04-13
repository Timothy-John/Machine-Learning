import numpy as np

#.....Loading Movie List from Text File.....
def get_Movie_List():
    movie_list = []
    with open('movie_ids.txt') as f:
        lines = f.readlines()
        for line in lines:
            idx, *movie_name = line.split(' ')
            movie_list.append(' '.join(movie_name).rstrip())
    # this code also works but more primitive
    """
    movie_list=[]
    list= open('movie_ids.txt')  # to display name of Rated Movie
    list=  list.read()
    list= list.split('\n')
    for lines in list:
        movie_list.append(lines.split(' ')[1:])
    """
    return movie_list

