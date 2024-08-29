#!/usr/bin/env python3
"""
Tune hyperparameters for an algorithm.

Usage:
    tune-algo.py [options] ALGO [DIR]

Options:
    -v, --verbose
        Increase logging verbosity.
    -r FILE, --record=FILE
        Record individual points to FILE.
    -o FILE
        Save parameters to FILE.
    -n N, --num-points=N
        Test N points in hyperparameter space [default: 60].
    --rmse
        Tune on RMSE instead of MRR.
    --points-only
        Print the points that would be used, without testing.
    --tags
        Include tags in training data.
    -d DIR, --test-data=DIR
        The test data directory [default: data/ml-25m-split].
    ALGO
        The algorithm to tune.
"""

import os
import sys
from pathlib import Path
import logging
from dataclasses import dataclass
import json
import csv
import io
from typing import Optional

from docopt import docopt
import pandas as pd
import numpy as np

from lenskit.datasets import MovieLens
from lenskit import batch, topn
from lenskit.algorithms import Recommender
from lenskit.util.parallel import invoker
from lenskit.util import Stopwatch
import seedbank

from dsci641 import algo_specs

_log = logging.getLogger("tune-algo")


@dataclass
class TuneContext:
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    test_users: np.ndarray
    algo_name: str
    tags: Optional[pd.DataFrame] = None
    metric: str = "MRR"


class PointRecord:
    metric: str
    record_file: io.TextIOBase | None
    record_writer: csv.DictWriter | None
    points: list

    def __init__(self, metric, space, file=None):
        self.metric = metric
        self.points = []
        if file:
            path = Path(file)
            path.parent.mkdir(exist_ok=True)
            self.record_file = open(file, "w", encoding="utf8")
            rcols = [name for (name, _dist) in space]
            rcols += [metric, "TrainTime", "RunTime"]
            self.record_writer = csv.DictWriter(self.record_file, rcols)
            self.record_writer.writeheader()

    def record(self, point):
        self.points.append(point)
        if self.record_writer:
            assert self.record_file is not None
            self.record_writer.writerow(point)
            self.record_file.flush()

    def close(self):
        if self.record_file:
            self.record_file.close()


def sample(space, state):
    "Sample a single point from a search space."
    return {name: dist.rvs(random_state=state) for (name, dist) in space}


def evaluate(ctx: TuneContext, point):
    "Evaluate an algorithm."
    _log.info("parameter point: %s", point)
    algo_mod = algo_specs.algorithms[ctx.algo_name]
    algo = algo_mod.from_params(**point)
    _log.info("evaluating %s", algo)

    extra = {}
    if ctx.tags is not None:
        extra["tags"] = ctx.tags

    if ctx.metric == "RMSE":
        ttime = Stopwatch()
        algo.fit(ctx.train_data, **extra)
        ttime.stop()
        _log.info("trained model in %s", ttime)

        rtime = Stopwatch()
        preds = batch.predict(algo, ctx.test_data)
        rtime.stop()

        errs = preds["prediction"] - preds["rating"]
        # assume missing values are completely off (5 star difference)
        errs = errs.fillna(5)
        val = np.sqrt(np.mean(np.square(errs)))
    else:
        algo = Recommender.adapt(algo)
        ttime = Stopwatch()
        algo.fit(ctx.train_data, **extra)
        ttime.stop()
        _log.info("trained model in %s", ttime)

        rtime = Stopwatch()
        recs = batch.recommend(algo, ctx.test_users, 5000)
        rtime.stop()
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank, k=5000)
        scores = rla.compute(recs, ctx.test_data, include_missing=True)
        val = scores["recip_rank"].fillna(0).mean()

    point.update(
        {
            ctx.metric: val,
            "TrainTime": ttime.elapsed(),
            "RunTime": rtime.elapsed(),
        }
    )
    return point


def main(args):
    level = logging.DEBUG if args["--verbose"] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    logging.getLogger("numba").setLevel(logging.INFO)

    seedbank.init_file("params.yaml")

    algo_name = args["ALGO"]
    _log.info("loading algorithm %s", algo_name)
    algo_mod = algo_specs.algorithms[algo_name]

    data = Path(args["--test-data"])
    _log.info("loading data from %s", data)
    train_data = pd.read_parquet(data / "tuning-train.parquet")
    test_data = pd.read_parquet(data / "tuning-test-filtered.parquet")
    test_users = test_data["user"].unique()

    ctx = TuneContext(train_data, test_data, test_users, algo_name)

    state = seedbank.numpy_random_state(seedbank.derive_seed(algo_name))

    if args["--tags"]:
        ml = MovieLens("data/ml-25m")
        ctx.tags = ml.tags

    if args["--rmse"]:
        _log.info("scoring predictions on RMSE")
        ctx.metric = "RMSE"

    record_fn = args["--record"]
    record = PointRecord(ctx.metric, algo_mod.space, record_fn)

    npts = int(args["--num-points"])
    _log.info("evaluating at %d points", npts)
    points = (sample(algo_mod.space, state) for _i in range(npts))
    procs = int(os.environ.get("TUNING_PROCS", "1"))
    if args["--points-only"]:
        for i, point in enumerate(points):
            _log.info("point %d: %s", i, point)
        return

    with invoker(ctx, evaluate, n_jobs=procs) as inv:
        for point in inv.map(points):
            _log.info("%s: %s=%.4f", point, ctx.metric, point[ctx.metric])
            record.record(point)

    points = sorted(
        record.points, key=lambda p: p[ctx.metric], reverse=(ctx.metric != "RMSE")
    )
    best_point = points[0]
    _log.info("finished in with %s %.3f", ctx.metric, best_point[ctx.metric])
    for p, v in best_point.items():
        _log.info("best %s: %s", p, v)

    record.close()

    fn = args.get("-o", None)
    if fn:
        _log.info("saving params to %s", fn)
        Path(fn).write_text(json.dumps(best_point))


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
