#!/usr/bin/env python3

"""
Run an algorithm.

Usage:
    run-algo.py [options] ALGO

Options:
    -v, --verbose
        Use verbose logging.
    -d NAME, --data=NAME
        Use data set NAME [default: ml-25m].
    --tags
        Use tag data in addition to ratings.
    -p PFX, --prefix=PREFIX
        Prefix output dirs with PFX.
    -j N, --procs N
        Use N processes for prediction / recommendation.
    --tuning
        Use temporal tuning data.
    --eval
        Use temporal evaluation data.
    --crossfold
        Use cross-validation split.
    --first-part N
        Start from partition N [default: 1].
    --params FILE
        Load parameters from FILE.
    --device DEV
        Use PyTorch device DEV instead of default.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd
from docopt import docopt
from dataclasses import dataclass, field

import seedbank
from lenskit.datasets import MovieLens
from lenskit.metrics.predict import rmse
from lenskit.batch import predict, recommend
from lenskit.util import clone
from lenskit.algorithms import Recommender, Algorithm
from lenskit.util.parallel import is_mp_worker

from dsci641.algo_specs import algorithms


_log = logging.getLogger("run-algo")
runs_base = Path("runs")

if is_mp_worker():
    # disable torch threading in worker proceses
    import torch

    torch.set_num_threads(1)

authors = pd.read_parquet('/content/drive/MyDrive/A3Data/gr-item-authors.parquet')
genres = pd.read_parquet('/content/drive/MyDrive/A3Data/gr-item-genres.parquet')
subjects = pd.read_parquet('/content/drive/MyDrive/A3Data/gr-item-subjects.parquet')

@dataclass
class DataSpec:
    key: str
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class AlgoSpec:
    name: str
    prefix: str | None

    model: Algorithm
    predicts_ratings: bool
    extra_data: dict[str, Any] = field(default_factory=dict)


def run_algo(algo: AlgoSpec, data: DataSpec):
    _log.info("preparing to evaluate %s", algo)
    # make a recommender
    model = clone(algo.model)
    model = Recommender.adapt(model)

    _log.info("training %s", algo.name)
    
    model.fit(data.train, **algo.extra_data)

    dname = f"{algo.prefix}-{algo.name}" if algo.prefix else algo.name
    out_dir = runs_base / dname
    _log.info("saving outputs to %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if algo.predicts_ratings:
        _log.info("generating predictions")
        preds = predict(model, data.test, n_jobs=n_jobs)
        err = rmse(preds["prediction"], preds["rating"])
        _log.info("finished with gi %4f", err)
        preds.to_parquet(out_dir / f"{data.key}-preds.parquet", index=False)

    _log.info("generating recommendations")
    recs = recommend(model, data.test["user"].unique(), 20, n_jobs=n_jobs)
    recs.to_parquet(out_dir / f"{data.key}-recs.parquet", index=False)


def data_specs(opts):
    data = opts["--data"]
    if opts["--eval"]:
        train = pd.read_parquet(f"/content/drive/MyDrive/{data}/a3-train-actions.parquet")
        test = pd.read_parquet(f"/content/drive/MyDrive/{data}/a3-dev-actions.parquet")
        yield DataSpec("eval", train, test)
    elif opts["--tuning"]:
        train = pd.read_parquet(f"data/{data}-split/tuning-train.parquet")
        test = pd.read_parquet(f"data/{data}-split/tuning-test-filtered.parquet")
        yield DataSpec("tuning", train, test)
    elif opts["--crossfold"]:
        start = int(opts["--first-part"])
        for i in range(start, 6):
            train = pd.read_parquet(f"data/{data}-crossfold/part{i}-train.parquet")
            test = pd.read_parquet(f"data/{data}-crossfold/part{i}-test.parquet")
            yield DataSpec(f"part{i}", train, test)



def main():
    global n_jobs
    opts = docopt(__doc__)
    # initialize logging
    level = logging.DEBUG if opts["--verbose"] else logging.INFO
    logging.basicConfig(level=level)
    # turn off numba debug, it's noisy
    logging.getLogger("numba").setLevel(logging.INFO)
    seedbank.init_file("/content/drive/MyDrive/DSCI641/params.yaml")
    data = opts["--data"]
    n_jobs = opts["--procs"]
    if n_jobs:
        n_jobs = int(n_jobs)

    algo_name = opts["ALGO"]
    _log.info("using algorithm %s", algo_name)
    algo_mod = algorithms[algo_name]
    pfn = opts.get("--params", None)
    if pfn:
        _log.info("using parameters from %s", pfn)
        params = json.loads(Path(pfn).read_text())
        algo = algo_mod.from_params(**params)
    else:
        algo = algo_mod.default()
    is_rating_predictor = getattr(algo_mod, "predicts_ratings", False)

    extra = {}
    if opts["--tags"]:
        _log.info("loading tag data")
        data = opts["--data"]
        extra["authors"] = authors
        extra["genres"] = genres
        extra["subjects"] = subjects
    dev = opts.get("--device", None)
    if dev:
        extra["device"] = dev
    elif "LK_TORCH_DEVICE" in os.environ:
        extra["device"] = os.environ["LK_TORCH_DEVICE"]

    algo = AlgoSpec(
        algo_name, opts.get("--prefix", None), algo, is_rating_predictor, extra
    )
    
    for data in data_specs(opts):
        run_algo(algo, data)

if __name__ == "__main__":
    main()

