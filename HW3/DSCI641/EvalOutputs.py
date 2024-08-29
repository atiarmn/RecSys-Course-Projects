# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluate Outputs
#
# This notebook evaluates outputs of the recommendation model(s).
#
# It evaluates two versions of most models: a default model, using some common hyperparameter settings, and a tuned model with hyperparameters selected by random search.  For comparison, this report also includes the performance of both model versions on both the evaluation and tuning data, so that we can detect cases where the hyperparamter tuning process overfit to the tuning data.

# %% [markdown]
# ## Setup
#
# Let's import some things:

# %%
from pathlib import Path
import re
import logging
from tqdm.notebook import tqdm

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
from lenskit import topn
from lenskit.datasets import MovieLens
from lenskit.util.log import log_to_notebook

# %% [markdown]
# Set up some logging:

# %%
_log = logging.getLogger("eval-notebook")
log_to_notebook()
tqdm.pandas()

# %% [markdown]
# ## Load Data
#
# What data set?

# %% tags=["parameters"]
data = "ml-25m"

# %% [markdown]
# ### Test Data
#
# We're going to start by loading the test data:

# %%
ml = MovieLens(f"data/{data}")
split_dir = Path(f"data/{data}-split")
run_dir = Path("runs")

# %%
movies = ml.movies
movies.head()

# %% [markdown]
# Scan for and load test data:

# %%
ev_test_ratings = pd.read_parquet(split_dir / "eval-test-filtered.parquet")
tu_test_ratings = pd.read_parquet(split_dir / "tuning-test-filtered.parquet")


# %% [markdown]
# ### Outputs
#
# Now let's load prediction & recommendation data.
#
# A regular expression will parse file names, where the algorithm names and variants are stored:

# %%
adn_re = re.compile(r"^ml-25m-(?P<variant>\w+)-(?P<algo>[A-Za-z-]+)")
fn_re = re.compile(r"(?P<key>\w+)-(?:recs|preds)")

# %% [markdown]
# Load predictions:

# %%
pred_frames = {}
for file in run_dir.glob(f"{data}-*/*-preds.parquet"):
    _log.info("loading %s", file)
    dmatch = adn_re.match(file.parent.name)
    var = dmatch.group("variant")
    algo = dmatch.group("algo")
    fmatch = fn_re.match(file.name)
    part = fmatch.group("key")
    df = pd.read_parquet(file)
    pred_frames[(algo, var, part)] = df

# %%
preds = pd.concat(pred_frames, names=["algo", "variant", "part"])
preds = preds.reset_index(["algo", "variant", "part"]).reset_index(drop=True)
preds.head()

# %% [markdown]
# And now the recommendations:

# %%
rec_frames = {}
for file in run_dir.glob(f"{data}-*/*-recs.parquet"):
    _log.info("loading %s", file)
    dmatch = adn_re.match(file.parent.name)
    var = dmatch.group("variant")
    algo = dmatch.group("algo")
    fmatch = fn_re.match(file.name)
    part = fmatch.group("key")
    df = pd.read_parquet(file)
    rec_frames[(algo, var, part)] = df

# %%
recs = pd.concat(rec_frames, names=["algo", "variant", "part"])
recs = recs.reset_index(["algo", "variant", "part"]).reset_index(drop=True)
recs.head()

# %% [markdown]
# ## Prediction Accuracy
#
# Now, let's compute the per-user RMSE of each algorithm.

# %%
preds["sqerr"] = np.square(preds["rating"] - preds["prediction"])
user_rmse = preds.groupby(["algo", "variant", "part", "user"])["sqerr"].mean()
user_rmse = user_rmse.to_frame("RMSE").reset_index()

# %% [markdown]
# We'll now print a tabular summary of RMSE:

# %%
user_rmse.groupby(["algo", "variant", "part"])["RMSE"].mean().unstack().unstack()

# %% [markdown]
# Plot the RMSEs (remember, **lower is better**):

# %%
sns.catplot(user_rmse, x="algo", y="RMSE", hue="variant", col="part", kind="bar")
plt.show()

# %% [markdown]
# ## Recommendation Utility
#
# Now let's evaluate the top-*N* recommendations.

# %%
rla = topn.RecListAnalysis()
rla.add_metric(topn.hit, k=20)
rla.add_metric(topn.recip_rank, k=20)
rla.add_metric(topn.ndcg, k=20)
ev_user_scores = rla.compute(
    recs[recs["part"] == "eval"], ev_test_ratings.drop(columns=["rating"]),
    include_missing=True
)
tu_user_scores = rla.compute(
    recs[recs["part"] == "tuning"], ev_test_ratings.drop(columns=["rating"]),
    include_missing=True
)
user_scores = pd.concat(
    [
        ev_user_scores.fillna(0).reset_index(),
        tu_user_scores.fillna(0).reset_index(),
    ]
).rename(
    columns={
        "recip_rank": "MRR",
        "hit": "HitRate",
        "ndcg": "NDCG",
    },
)

# %%
user_scores

# %% [markdown]
# In order to plot the results, we need a *tall* data frame:

# %%
us_tall = user_scores.melt(
    id_vars=["algo", "variant", "part", "user"],
    value_vars=["MRR", "HitRate", "NDCG"],
    var_name="Metric",
)


# %% [markdown]
# Now we chart the top-*N* performance across our three metrics:

# %%
g = sns.catplot(
    us_tall,
    y="algo",
    x="value",
    col="Metric",
    row="part",
    hue="variant",
    kind="bar",
    sharex=False,
)
g.set_titles("{row_name} {col_name}")
g.set_xlabels(None)
plt.show()

# %% [markdown]
# We want a tabular summary too.

# %%
means = us_tall.groupby(['algo', 'variant', 'part', 'Metric'])['value'].mean()
means = means.unstack().unstack().unstack()
means.style.highlight_max()

# %% [markdown]
# We see that prior to tuning, SampLMF is the top performer across the board, but I-MF takes over after tuning the hyperparamters.

# %%
