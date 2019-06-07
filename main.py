#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import prediction
import recommender
import utilities
import truncated_map
# import numpy as np

def evaluate_algorithms(training_file, testing_file, algorithm):
    # users = list(utilities.load_users("data/kaggle_users.txt"))[:100]

    s_u = utilities.song_to_users_min(training_file) #dict songs:{users}
    #
    # for algorithm in algorithms:
    pr = prediction.ItemBasedPrediction(s_u, _sim=0) #also 1. #model right here...
    u_s = utilities.user_to_songs(testing_file)


    recommendations = []
    rec = recommender.Recommender(algorithm, pr) #song_ordered is the algorithm. baseline = popularity.
    # print('making', len(u_s.keys()), 'recs')
    # print(list(u_s.keys())[:10], list(u_s.values())[:10])
    for i, user in enumerate(u_s.keys()):
        res = rec.recommend(user, u_s) #these are the recommended songs.
        # utilities.save_recommendations(res, "data/result3.txt")
        recommendations.append(res)
        if i % 20 == 0:
            print('recommended', i, 'so far')

    mAP_value = truncated_map.mean_average_precision(users, recommendations)
    print(len(users), 'users', mAP_value, 'mAP_value')
    #     #do algorithm
    #
    #     for metric in metrics: #mAP, possible others?
    #         #evaluate


    # scores = []
    # for split in generate_splits(dataset, folds):
    #     train, test = split
    #     actual = [v[-1] for v in test]
    #     test = [v[:-1] + [None] for v in test]
    #     predicted = algorithm(train, test, *args)
    #     result = metric(actual,predicted)
    #     scores.append(result)
    # return scores

def main():
    # training_file = "data/year1_test_triplets_visible.txt"
    training_file = "data/kaggle_visible_evaluation_triplets.txt"
    testing_file = "data/year1_valid_triplets_hidden.txt"
    songs_ordered = utilities.sort_dict(utilities.count_songs(training_file))
    evaluate_algorithms(training_file, testing_file, songs_ordered) #took out metric: mAP (goes here)

def main2():
    # usage python main [userId] [cosine|prod](cosine by default)
    # opt : cosine => cosine-based similarity
    # opt : prob => conditional probability-based similarity
    user_id, opt = sys.argv[1:]
    user_id = int(user_id)
    opt = str(opt)

    print("Building recommendation for user : {:d}".format(user_id))
    # TRIPLETS FILE
    file_tt = "data/year1_test_triplets_visible.txt"
    # We used kaggle evaluation triplets since they are different from our training
    file_tev = "data/year1_test_triplets_hidden.txt"

    print("Load Users from users list")
    users = list(utilities.load_users("data/kaggle_users.txt"))

    print("Ordering song by popularity") #this is an algorithm
    songs_ordered = utilities.sort_dict(utilities.count_songs(file_tt))

    # print("Load unique Users indexes")
    # uniq_users = utilities.unique_users(file_tt)
    #
    # print("Enumerate User with indexes")
    # u_i = dict()
    # for i, u in enumerate(uniq_users):
    #     u_i[u] = i

    print("Loading MSD training triplets")
    s_u = utilities.song_to_users_min(file_tt)

    # print("Converting Users indexes")
    # for s in s_u:
    #     s_set = set()
    #     for u in s_u[s]:
    #         s_set.add(u_i[u])
    #     s_u[s] = s_set
    #
    # del u_i

    # creating prediction
    if opt == "cosine":
        opt = 0
        print("Using cosine based similarity prediction")
    if opt == "prob":
        opt = 1
        print("Using cond-prob based similarity prediction")

    pr = prediction.ItemBasedPrediction(s_u, _sim=opt)
    # pr = prediction.UserBasedPrediction( )

    print("Loading users evaluation triplets")
    # u_s : dict of users with the set of songs the user has listened to (user vector)
    u_s = utilities.user_to_songs(file_tev)

    rec = recommender.Recommender(songs_ordered, pr)

    res = rec.recommend(users[user_id], u_s)
    utilities.save_recommendations(res, "data/result3.txt")

    # Evaluation
    # yp = np.array([])
    # precision_recall_fscore_support(yt, yp, average="weighted")
    # utilities.print_results()


if __name__ == "__main__":
    main()
