"""
Train and recommend with an algorithm.

Usage:
    train-recommend.py [options] ALGO

Options:
    --users N
        Test on N users [default: 1000]
    --data DATA
        Train on dataset DATA [default: ml-20m]
"""
import logging
import os
from lenskit.crossfold import sample_users, SampleN
from lenskit.datasets import MovieLens, ML10M
from lenskit.algorithms import Recommender
from lenskit.batch import recommend, predict
from lenskit.topn import RecListAnalysis, ndcg, hit, precision, recall
from lenskit.metrics.predict import rmse
from lenskit.util import init_rng
from docopt import docopt
import torch
import numpy as np
import pandas as pd
import seedbank

from algo_defs import algorithms, pred_algos

_log = logging.getLogger('train-recommend')


def main():
    logging.getLogger('numba').setLevel(logging.INFO)
    seedbank.init_file("/content/drive/MyDrive/Recsys_final/params.yaml", "/content/drive/MyDrive/Recsys_final/train-model")
    logging.basicConfig(level=logging.INFO)
    init_rng(20200306)
    torch.manual_seed(20200306)

    opts = docopt(__doc__)
    n_users = int(opts['--users'])
  
    n_parts = 5
    ndcgs = []
    hits = []
    recalls = []
    precisions = []
    tags = pd.read_parquet("/content/drive/MyDrive/Recsys_final/data/item-tags.parquet")
    for part in range(n_parts):
        train_file = f'/content/drive/MyDrive/Recsys_final/data/split/part{part}-train.parquet'
        test_file = f'/content/drive/MyDrive/Recsys_final/data/split/part{part}-test.parquet'

        _log.info('reading train data from %s', train_file)
        train = pd.read_parquet(train_file)
        _log.info('reading test data from %s', test_file)
        test = pd.read_parquet(test_file)

        _log.info('training model %s for part %d', opts['ALGO'], part)
        algo = algorithms[opts['ALGO']]
        model = Recommender.adapt(algo)

        dev = opts.get("--device", None)
        if dev is None:
            dev = os.environ.get("LK_TORCH_DEVICE", None)

        _log.info("training %s", opts['ALGO'])

        model.fit(
            train,
            tags=tags,
            device=dev,
        )

        _log.info('generating recs for part %d', part)
        users = test['user'].unique()
        recs = recommend(model, users, 20, n_jobs=1)

        _log.info('received %s recommendations for part %d', len(recs), part)

        rla = RecListAnalysis(n_jobs=1)
        rla.add_metric(ndcg)
        rla.add_metric(precision)
        rla.add_metric(recall)
        rla.add_metric(hit)
        res = rla.compute(recs, test)
        res = res.reset_index()
        res['Algorithm'] = opts['ALGO']
        _log.info('res: ', res.head())
        _log.info('average NDCG: %.4f', res['ndcg'].mean())
        _log.info('average hit: %.4f', res['hit'].mean())
        _log.info('average recall: %.4f', res['recall'].mean())
        _log.info('average precision: %.4f', res['precision'].mean())
        ndcgs.append(res['ndcg'].mean())
        hits.append(res['hit'].mean())
        recalls.append(res['recall'].mean())
        precisions.append(res['precision'].mean())
        
        if opts['ALGO'] in pred_algos:
            preds = predict(algo, test, n_jobs=1)
            p_rmse = rmse(preds['prediction'], preds['rating'])
            _log.info('global RMSE for part %d: %.4f', part, p_rmse)

    # Concatenate all metrics and save to a parquet file
    algorithm = opts['ALGO']
    all_metrics = {'Algo': algorithm, 'ndcg':np.mean(ndcgs), 'hit':np.mean(hits), 'recall':np.mean(recalls), 'precision':np.mean(precisions)}
    met = pd.DataFrame([all_metrics])
    met.to_parquet(f'/content/drive/MyDrive/Recsys_final/outs/rec-metrics-{algorithm}-bpr.parquet')


if __name__ == '__main__':
    main()