#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import prediction
import recommender
import utilities
import truncated_map

from multiprocessing import Pool

# import numpy as np

def parallel_rec_worker(user, rec, u_s):
    # print('working on rec')
    song_recs = rec.recommend(user, u_s)
    return [user, song_recs]

#This callback writes a user and its recommended songs to a file.
#Name the file the the method of prediction being used. IE. Popular, UserBased, ItemBased
def log_result(result):
    user, songs_recs = result[0], result[1]
    # This is called whenever pool.apply_async(i) returns a result.
    # called only by the main process, not the pool workers.
    with open('UserBasedRecs', 'a') as recs:
        recs.write(user + ' ' + ' '.join(songs_recs) +'\n')


def evaluate_algorithm(training_file, testing_file):
    s_u = utilities.song_to_users(training_file) #dict songs:{users}

    # pr = prediction.ItemBasedPrediction(s_u, _sim=0)
    pr = prediction.UserBasedPrediction(s_u)
    rec = recommender.Recommender(all_songs, pr)

    u_s = utilities.user_to_songs(testing_file)

    pool = Pool(4)
    t = time.time()
    for user in u_s.keys():
        pool.apply_async(parallel_rec_worker, args = (user,rec,u_s,), callback = log_result)

    print('finished applying')
    pool.close()
    print('about to join...')
    pool.join()
    print('finished jobs...')


def main():
    # training_file = "data/year1_test_triplets_visible.txt"
    training_file = "data/kaggle_visible_evaluation_triplets.txt"
    testing_file = "data/year1_valid_triplets_hidden.txt"
    songs_ordered = utilities.sort_dict(utilities.count_songs(training_file))
    evaluate_algorithm(training_file, testing_file, songs_ordered) #took out metric: mAP (goes here)


if __name__ == "__main__":
    main()
