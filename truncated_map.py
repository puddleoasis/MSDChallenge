#!/usr/bin/python
# -*- coding: utf-8 -*-

""" This scripts provides functions for computing the Mean Average Precision
 - reference to the formulas used in this script :
 The Million Song Dataset Challenge - Thierry Bertin-Mahieux, Brian McFee """

import numpy as np
import utilities


def average_precision(rec_songs, actual_songs, k=999):
    """ This function computes the average precision at each recall point
    - param :
             rec_songs    : list of recommended songs
             actual_songs : list of songs the user listened to
    """
    np = len(actual_songs)
    # print "np:", np
    mapr_user = 0.0
    nc = 0.0
    for j, s in enumerate(rec_songs):
        if j >= k:
            break
        if s in actual_songs:
            nc += 1.0
            mapr_user += nc / (j+1)
    mapr_user /= min(np, k)
    return mapr_user



def mean_average_precision(training_m, testing_m):
    """ This function computes the mean average precision
    - reference to the mAP formula:
    The Million Song Dataset Challenge - Thierry Bertin-Mahieux, Brian McFee
    - param :
           training_m   : dict of all users to predicted songs
           testing_m    : dict of user to actual songs
    - return:
           the mean average precision
    """
    return np.mean([average_precision(training_m[key], testing_m[key]) for key in testing_m.keys()])


def test_predictions(training_file, testing_file, k=500):
    testing_matrix = utilities.user_to_songs(testing_file)

    counted_songs = utilities.count_songs(training_file)
    songs_ordered = utilities.sort_dict(counted_songs)
    top_songs = songs_ordered[:k]

    train_matrix = {}

    recFiles = ['UserBasedRecs', 'PopularityBasedRecs', 'ItemBasedRecs']
    for file in recFiles:
        with open('generatedRecommendations/'+file, 'r') as recd:
            for line in recd:
                d = line.strip().split()
                u, s = d[0], d[1:]
                train_matrix[u] = s
        canonical_users = []
        with open('data/kaggle_users.txt', 'r') as f:
            canonical_users = map(lambda line: line.strip(), f.readlines())
            for cu in canonical_users:
                if cu not in train_matrix.keys():
                    train_matrix[cu] = top_songs
        mAP = mean_average_precision(train_matrix,testing_matrix)
        print(file, testing_file, mAP)


def main():
    test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/year1_valid_triplets_visible.txt") #partial
    test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/year1_test_triplets_visible.txt")  #full
    test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/year1_valid_triplets_hidden.txt")  #partial
    test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/kaggle_visible_evaluation_triplets.txt") #full

    # test_precision("data/year1_test_triplets_hidden.txt", "data/year1_test_triplets_hidden.txt") #full
    # test_precision("data/year1_test_triplets_visible.txt", "data/year1_test_triplets_visible.txt")  #partial
    # test_precision("data/year1_valid_triplets_hidden.txt", "data/year1_valid_triplets_hidden.txt")  #full
    # test_precision("data/year1_valid_triplets_visible.txt", "data/year1_valid_triplets_visible.txt") #partial
    # test_precision("data/kaggle_visible_evaluation_triplets.txt", "data/kaggle_visible_evaluation_triplets.txt") #full
main()
