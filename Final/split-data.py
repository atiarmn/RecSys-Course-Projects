import logging
from pathlib import Path
import pandas as pd
from lenskit.datasets import MovieLens
from lenskit import crossfold as xf

_log = logging.getLogger('split-data')


def main():
    logging.basicConfig(level=logging.INFO)
    _log.info('reading ratings')
    ratings = pd.read_csv("/content/drive/MyDrive/Recsys_final/data/RAW_interactions.csv")
    ratings.rename(columns={"user_id": "user", "recipe_id": "item"},inplace = True)
    split = Path('/content/drive/MyDrive/Recsys_final/data/split')
    split.mkdir(exist_ok=True)
    min_ratings = 5
    user_counts = ratings['user'].value_counts()
    active_users = user_counts[user_counts >= min_ratings].index
    filtered_ratings = ratings[ratings['user'].isin(active_users)]

    for i, (train, test) in enumerate(xf.partition_users(filtered_ratings, 5, xf.SampleN(5))):
        _log.info('writing partition %d', i)
        train.to_parquet(split / f'part{i}-train.parquet', index=False)
        test.to_parquet(split / f'part{i}-test.parquet', index=False)


if __name__ == '__main__':
    main()