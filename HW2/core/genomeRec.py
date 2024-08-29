
import logging
import numpy as np
import pandas as pd

from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings
from lenskit.util import Stopwatch

# I want a logger for information
_log = logging.getLogger(__name__)


class GenomeRec(Predictor):

    def fit(self, ratings, genome):
       
        timer = Stopwatch()

        _log.info('[%s] training started', timer)
        genome = genome.pivot(index='movieId', columns='tagId', values='relevance')
        gnorms = genome.apply(np.linalg.norm, axis=1)
        norm_genome = genome.multiply(np.reciprocal(gnorms), axis=0)
        movie_ids = norm_genome.index

        _log.info('[%s] training finished, saving %d results', timer)
        self.item_index_ = movie_ids
        self.genome_ = norm_genome
        self.ratings_ = ratings
    

    def predict_for_user(self, user, items, ratings=None):
       
        if ratings is None:
            ratings = self.ratings_
        u_rates = ratings[ratings['user'] == user]
        masked_ratings = u_rates[u_rates['item'].isin(self.item_index_)]
        u_tags = self.genome_.loc[masked_ratings['item']]
    
        u_tags_vectors = np.average(u_tags, axis=0, weights=masked_ratings['rating'])
        
        tags = pd.DataFrame(u_tags_vectors.reshape(1,-1),columns = self.genome_.columns, index = [user])
    
        tagnorms = tags.apply(np.linalg.norm, axis=1)
        user_tags = tags.multiply(np.reciprocal(tagnorms), axis=0)
        similarity_scores = self.genome_ @ user_tags.T

        results = pd.Series(similarity_scores[user])

        return results.reindex(items)  # fill in missing values with nan
