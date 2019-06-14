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


def test_predictions(training_file, testing_file, k=10):
    testing_matrix = utilities.user_to_songs(testing_file)

    counted_songs = utilities.count_songs(training_file)
    songs_ordered = utilities.sort_dict(counted_songs)
    top_songs = songs_ordered[:k]

    train_matrix = {}

    # Use main.py to generate either UserBasedRecs or ItemBasedRecs. #
    # Then uncomment the code below an ensure the path is correct.

    # The User and Item based recommendations take a long time to generate.
    # Writting them to a file will allow us to test the results with a different
    # metric in the future.
    recommendationFiles = ['PopularityBasedRecs','ItemBasedPrediction']
    file = recommendationFiles[0]
    with open('generatedRecommendations/'+file, 'r') as recd:
        for line in recd:
            d = line.strip().split()
            user, songs = d[0], d[1:k]
            train_matrix[user] = songs



    #This code assigns the most popular songs to each user:
    # (We discuss this in the Recommended Experiments Section of the paper)
    canonical_users = []
    # add any user that might appear in our testing data.
    with open('data/kaggle_users.txt', 'r') as f:
        canonical_users = map(lambda line: line.strip(), f.readlines())
        for cu in canonical_users:
            if cu not in train_matrix.keys():
                train_matrix[cu] = top_songs
    mAP = mean_average_precision(train_matrix,testing_matrix)
    print('testing file:', testing_file, 'mAP=', mAP, 'k=',k)


def main():
    for i in [10, 25, 50, 100, 250, 500]:
        test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/year1_valid_triplets_visible.txt", k=i) #partial
        # test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/year1_test_triplets_visible.txt", k=i)  #full
        # test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/year1_valid_triplets_hidden.txt", k=i)  #partial
        # test_predictions("data/kaggle_visible_evaluation_triplets.txt", "data/kaggle_visible_evaluation_triplets.txt", k=i) #full
main()
