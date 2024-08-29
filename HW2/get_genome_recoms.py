# -*- coding: utf-8 -*-


from core import genomeRec
from lenskit import Recommender
from core import datasets
from lenskit import batch

num_recommendations = 50
genomeRec = genomeRec.GenomeRec()
genome_data = getattr(datasets, 'genome_data')

for partition in range(5):
    ratings = getattr(datasets, f'train_{partition+1}')
    test = getattr(datasets, f'test_{partition+1}')
    # wrap the algorithm instance in recommender
    recommender = Recommender.adapt(genomeRec)

    recommender.fit(ratings = ratings, genome = genome_data)


    # generate recommendations for all users
    print("Generating recommendations...")
    test = test['user'].unique()
    recs = batch.recommend(recommender, test, num_recommendations)
    recs.to_csv(f'results/partition_{partition+1}-genome-recs.csv', index=False)

