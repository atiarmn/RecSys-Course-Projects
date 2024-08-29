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
# # MovieLens Data Overview
#
# This documents some basic statistics for the ML data.

# %% [markdown]
# ## Setup
#
# Load libraries:

# %%
from pathlib import Path

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
from humanize import metric

# %% [markdown]
# And load the data:

# %%
from lenskit.datasets import MovieLens
dataset = 'ml-25m'
ml = MovieLens(f'data/{dataset}')

# %% [markdown]
# Output location:

# %%
out_dir = Path(f'data/{dataset}-split')
out_dir.mkdir(exist_ok=True)


# %% [markdown]
# ## Basic Statistics

# %%
def summarize_ratings_frame(df):
    row = {
        'Ratings': len(df),
        'Users': df['user'].nunique(),
        'Items': df['item'].nunique(),
    }
    row['Density'] = len(df) / (row['Users'] * row['Items'])
    return row

def summarize_rating_sets(dfs):
    names = []
    rows = []
    for name, df in dfs.items():
        names.append(name)
        rows.append(summarize_ratings_frame(df))

    return pd.DataFrame.from_records(rows, index=names).style.format({
        'Ratings': metric,
        'Users': metric,
        'Items': metric,
        'Density': lambda d: '{:.3f}%'.format(d*100)
    })


# %%
ratings = ml.ratings
summarize_rating_sets({'Source': ratings})

# %% [markdown]
# How is user activity distributed?

# %%
sns.ecdfplot(ratings['user'].value_counts(), log_scale=True)
plt.xlabel('# of Ratings')
plt.ylabel('Cumulative Proportion of Users')
plt.title('User Activity Distribution')
plt.show()

# %% [markdown]
# And item popularity?

# %%
sns.ecdfplot(ratings['item'].value_counts(), log_scale=True)
plt.title('Item Popularity Distribution')
plt.xlabel('# of Ratings')
plt.ylabel('Cumulative Proportion of Items')
plt.show()

# %% [markdown]
# ## Data over Time
#
# We're now going to look at the data *over time*.

# %%
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['timestamp'].describe()

# %%
rts = ratings.set_index('timestamp')
rts.sort_index(inplace=True)

# %% [markdown]
# Now that we have a timestamp column, we can *resample* the data and plot activity by month:

# %%
rts.resample('1M')['rating'].count().plot.line()
plt.xlabel('Time')
plt.ylabel('Ratings per Month')
plt.title("Rating Volume over Time")
plt.show()

# %% [markdown]
# What about unique users?

# %%
rts.resample('1M')['user'].nunique().plot.line()
plt.xlabel('Time')
plt.ylabel('Unique Users per Month')
plt.title("Monthly Unique Users")
plt.show()

# %% [markdown]
# ## Data Split
#
# Let's make a temporal split of training and test data.  We'll use Q4 2018 for our evaluation, and Q4 2017 for algorithm tuning.  This means we need *two* train-test splits:

# %%
ev_test = ratings.query("timestamp >= '2018-10-01' and timestamp < '2019-01-01'")
ev_train = ratings.query("timestamp < '2018-10-01'")
ev_test['timestamp'].describe()

# %% [markdown]
# Let's look at the distribution of **training profile** sizes:

# %%
ev_test_users = ev_test.groupby('user')['rating'].agg(['count', 'mean'])
ev_train_users = ev_train.groupby('user')['rating'].agg(['count', 'mean'])
ev_test_users = ev_test_users.join(ev_train_users, how='left', rsuffix='_train')
ev_test_users['count_train'].fillna(0, inplace=True)
ev_test_users.head()

# %%
ev_test_users['count_train'].describe()

# %% [markdown]
# We have 5211 test users, of which at least 25% have no training ratings. Let's count the no-training users:

# %%
np.sum(ev_test_users['count_train'] == 0)

# %% [markdown]
# This will only leave us with about 3400 users to test with, but we can work with that for demonstration purposes.  We're going to filter the test data
# so we only include users who are in the training data:

# %%
ev_test_filtered = ev_test[ev_test['user'].isin(ev_train_users.index)]

# %% [markdown]
# And now look at some data statistics:

# %%
summarize_rating_sets({
    'Source': ratings,
    'Training': ev_train,
    'Eval (unfiltered)': ev_test,
    'Eval (filtered)': ev_test_filtered
})

# %%
ev_train.to_parquet(out_dir / 'eval-train.parquet')
ev_test.to_parquet(out_dir / 'eval-test.parquet')
ev_test_filtered.to_parquet(out_dir / 'eval-test-filtered.parquet')

# %% [markdown]
# ## Tuning Data
#
# Now we're going to create the "tune train" and "tune test" data sets for hyperparameter tuning:

# %%
tu_test = ratings.query("timestamp >= '2017-10-01' and timestamp < '2018-01-01'")
tu_train = ratings.query("timestamp < '2017-10-01'")
tu_test['timestamp'].describe()

# %% [markdown]
# And filter the users:

# %%
tu_test_users = tu_test.groupby('user')['rating'].agg(['count', 'mean'])
tu_train_users = tu_train.groupby('user')['rating'].agg(['count', 'mean'])
tu_test_users = tu_test_users.join(tu_train_users, how='left', rsuffix='_train')
tu_test_users['count_train'].fillna(0, inplace=True)
tu_test_filtered = tu_test[tu_test['user'].isin(tu_train_users.index)]

# %%
tu_train.to_parquet(out_dir / 'tuning-train.parquet')
tu_test.to_parquet(out_dir / 'tuning-test.parquet')
tu_test_filtered.to_parquet(out_dir / 'tuning-test-filtered.parquet')

# %% [markdown]
# ## Final Data Summary

# %%
summarize_rating_sets({
    'Source': ratings,
    'Training': ev_train,
    'Eval (unfiltered)': ev_test,
    'Eval (filtered)': ev_test_filtered,
    'Tune Training': tu_train,
    'Tune Test (unfiltered)': tu_test,
    'Tune Test (filtered)': tu_test_filtered,
})
