#!/usr/bin/python
# -*- coding: utf-8 -*-

def get_song_id(id):
    """ This function get the echonest if of the song from the songs file given the id of the song"""
    with open("data/kaggle_songs.txt") as file:
        for line in file:
            line = line.strip().split()
            if id == line[1]:
                return line[0]


def count_songs(file):
    """ This function counts the number of users that listens to each song """
    s = dict()
    with open(file, "r") as f:
        for line in f:
            _, song, _ = line.strip().split('\t')
            try:
                s[song] += 1
            except KeyError:
                s[song] = 1
    return s


def count_users(file):
    """ This functions count the users given a triplet input file """
    u = dict()
    with open(file, "r") as f:
        for line in f:
            user, _, _ = line.strip().split('\t')
            try:
                u[user] += 1
            except KeyError:
                u[user] = 1
    return u


def sort_dict(d):
    """ This function returns the given dictionary d sorted by its keys"""
    return sorted(d.keys(), key=lambda s: d[s], reverse=True)


def song_to_users(file):
    """ This function loads user,song,play_count triplets and returns the dictionary of song with set of users that
    listens to that song """
    d = dict()
    with open(file, "r") as f:
        for line in f:
            user, song, _ = line.strip().split('\t')
            try:
                d[song].add(user)
            except KeyError:
                d[song] = {user}
    return d


def user_to_songs(file):
    """ This functions loads user,song,play_count triplets and returns the dictionary of users with the set of songs
    the user has listened to """
    d = dict()
    with open(file, "r") as f:
        for line in f:
            user, song, _ = line.strip().split('\t')
            try:
                d[user].add(song)
            except KeyError:
                d[user] = {song}
    return d


def load_users(file):
    """ This function loads users from users file and returns list of users"""
    with open(file, "r") as f:
        u = map(lambda line: line.strip(), f.readlines())
    return u

def unique_users_dict(file):
    """ This function returns a dict of unique user and a corresponding unique int
    - param:
          file : training file
    """
    u = dict()
    with open(file, "r") as f:
        for i, line in enumerate(f):
            user, _, _ = line.strip().split('\t')
            if user not in u.keys():
                u[user] = i
    return u

def unique_users(file):
    """ This function returns a set of unique user
    - param:
          file : training file
    """
    u = set()
    with open(file, "r") as f:
        for line in f:
            user, _, _ = line.strip().split('\t')
            if user not in u:
                u.add(user)
    return u


def save_recommendations(r, file):
    """ This function saves recommendation given in argument into file
    - param :
             r     list of songs to save
             file  output file
    """
    print("Saving recommendations")
    f = open(file, "w")
    for r_songs in r:
        out_line = [str(r_songs), '\n']
        f.writelines(out_line)
    f.close()

def song_to_users_min(file):
    """ This functions loads user,song,play_count triplets and returns the dictionary of users with the set of songs
        the user has listened to
        - param
            file : training file
    """
    with open(file, "r") as f:
        songs_user = dict()
        u_users = dict()
        u_songs = dict()
        for i, line in enumerate(f):
            user, song, _ = line.strip().split('\t')

            if user not in u_users:
                u_users[user] = i
            if song not in u_songs:
                u_songs[song] = i
                # songs_user[i] = {u_users[user]}
                songs_user[song] = {u_users[user]}
            else:
                songs_user[song].add(u_users[user])
    return songs_user
