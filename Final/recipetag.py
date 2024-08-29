"""
Implementation of a PyTorch-based recommender.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import NamedTuple
from torch._C import TracingState
from tqdm.auto import tqdm
import math
import numpy as np
import pandas as pd

from csr import CSR

import torch
from torch import nn
from torch.optim import AdamW
from torch.linalg import vecdot
import torch.nn.functional as F

from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util

# I want a logger for information
_log = logging.getLogger(__name__)


# named tuples are a quick way to make classes that are tuples w/ named fields
class ItemTags(NamedTuple):
    """
    Item tags suitable for input to an EmbeddingBag.  This is used for both
    genres and tags.
    """

    tag_ids: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_items(cls, matrix, items):
        if isinstance(items, torch.Tensor):
            items = items.cpu().numpy()
        # pick_rows gets a subset of the CSR with the specified rows.
        # its row pointers and column indexes are exactly what the embedding
        # matrix needs.
        tmat = matrix.pick_rows(items.ravel(), include_values=False)
        # make convert to numpy, but make sure things are sized correctly
        return cls(
            torch.from_numpy(tmat.colinds[: tmat.nnz]),
            torch.from_numpy(tmat.rowptrs[:-1]),
        )

    def to(self, dev):
        return ItemTags(self.tag_ids.to(dev), self.offsets.to(dev))


class Batch(NamedTuple):
    "Representation of a single batch."

    "The user IDs (B,1)"
    users: torch.Tensor
    "The item IDs (B,2); column 0 is positive, 1 negative"
    items: torch.Tensor

    "The batch size"
    size: int

    def to(self, dev):
        "move this batch to a device"
        return self._replace(
            users=self.users.to(dev),
            items=self.items.to(dev),
        )


@dataclass
class SampleEpochData:
    """
    Permuted data for a single epoch of sampled training.
    """

    data: recipeTrainData
    permutation: np.ndarray

    @property
    def n_samples(self):
        return self.data.n_samples

    @property
    def batch_size(self):
        return self.data.batch_size

    @property
    def batch_count(self):
        return math.ceil(self.n_samples / self.batch_size)

    def batch(self, batchno: int) -> Batch:
        start = batchno * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        size = end - start

        # find the rows for this sample
        rows = self.permutation[start:end]

        # get user IDs
        uv = self.data.uinds[rows].reshape((size, 1))

        # we will get a pair of items for each user - initialize array
        iv = np.empty((size, 2), dtype="int32")
        # get positive item IDs
        iv[:, 0] = self.data.matrix.colinds[rows]
        # get negative item IDs
        # it only works with vectors, not matrices, of user ids, so get column
        iv[:, 1], scts = sampling.neg_sample(
            self.data.matrix, uv[:, 0], sampling.sample_unweighted
        )
        # quick debug check
        if np.max(scts) > 7:
            _log.info("%d triples took more than 7 samples", np.sum(scts > 5))

        # we're done, send to torch and return
        return Batch(torch.from_numpy(uv), torch.from_numpy(iv), size)


@dataclass
class recipeData:
    """
    Capture data about recipes that is saved after training.
    """

    # user and item indices
    users: pd.Index
    items: pd.Index

    # item-tag matrix
    tag_mat: CSR

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    @property
    def n_tags(self):
        return self.tag_mat.ncols if self.tag_mat else 0

    def recipe_tags(self, recipes) -> ItemTags:
        return ItemTags.from_items(self.tag_mat, recipes)


@dataclass
class recipeTrainData:
    """
    Class capturing MF training data/context
    """

    # user and item indices
    users: pd.Index
    items: pd.Index

    matrix: CSR

    # consumption data
    r_users: np.ndarray
    r_items: np.ndarray

    batch_size: int

    @property
    def n_samples(self):
        return len(self.r_users)

    @property
    def batch_count(self):
        return math.ceil(self.n_samples / self.batch_size)

    def batch(self, rows):
        # get the ratings for this batch
        sz = len(rows)
        ub = self.r_users[rows]

        ib = np.empty((sz, 2), np.int32)
        ib[:, 0] = self.r_items[rows]
        ib[:, 1], jsc = sampling.neg_sample(self.matrix, ub, sampling.sample_unweighted)

        ub = ub.reshape((sz, 1))
        ub = torch.from_numpy(ub)
        ib = torch.from_numpy(ib)

        return Batch(ub, ib, sz)


class recipeNet(nn.Module):
    """
    Torch module that defines the hybrid matrix factorization model.

    Args:
        data(recipeData): the recipe metadata for content-based embeddings
        n_feats(int): the embedding dimension
        user_bias(bool): whether to include a user bias term
    """

    recipe_data: recipeData
    n_feats: int

    i_full = None

    def __init__(self, data: recipeData, n_feats: int, user_bias: bool = True):
        super().__init__()
        self.recipe_data = data
        self.n_feats = n_feats

        # user and item bias terms
        if user_bias:
            self.u_bias = nn.Embedding(data.n_users, 1)
        else:
            self.u_bias = None
        self.i_bias = nn.Embedding(data.n_items, 1)

        # user and item embeddings
        self.u_embed = nn.Embedding(data.n_users, n_feats)
        self.i_embed = nn.Embedding(data.n_items, n_feats)

        # tag embeddings
        if data.n_tags:
            self.a_embed = nn.EmbeddingBag(data.n_tags, n_feats)
        else:
            self.a_embed = None
        # rescale all initial values for better starting point
        # they started out as standard normals, those are pretty big
        if self.u_bias is not None:
            self.u_bias.weight.data.mul_(0.05)
            self.u_bias.weight.data.square()
        self.i_bias.weight.data.mul_(0.05)
        self.i_bias.weight.data.square()
        self.u_embed.weight.data.mul_(0.05)
        self.i_embed.weight.data.mul_(0.05)
        if self.use_tags:
            self.a_embed.weight.data.mul_(0.05)
       

    @property
    def device(self):
        return self.i_bias.weight.data.device

    @property
    def use_tags(self):
        return self.a_embed is not None

   
    def forward(self, users, items):
        ub, uvec = self._user_rep(users)
        ib, ivec = self._item_rep(items)

        score = ib + ub + vecdot(uvec, ivec)

        return score

    def _user_rep(self, ut):
        ub = 0.0
        if self.u_bias is not None:
            ub = self.u_bias(ut).reshape(ut.shape)

        uvec = self.u_embed(ut)

        return ub, uvec

    def _item_rep(self, it):
        ib = self.i_bias(it).reshape(it.shape)

        if self.i_full is not None:
            ivec = self.i_full(it)
        else:
            ivec = self.i_embed(it)
            if self.use_tags:
                tags = self.recipe_data.recipe_tags(it).to(self.device)
                avec = self.a_embed(tags.tag_ids, tags.offsets)
                ivec = ivec + avec.reshape(ivec.shape)
        
        return ib, ivec

    def compact(self, *, init_only=False):
        """
        Collapse item feature embeddings into integrated item embeddings
        for fast recommendations.
        """
        if init_only:
            self.i_full = nn.Embedding(self.recipe_data.n_items, self.n_feats)
            return

        iw = self.i_embed.weight.data
        n, k = iw.shape

        if self.a_embed is not None:
            amat = self.recipe_data.tag_mat
            ainput = torch.from_numpy(amat.colinds).to(self.device)
            aoffset = torch.from_numpy(amat.rowptrs[:-1]).to(self.device)
            aw = self.a_embed(ainput, aoffset)
            assert aw.shape == iw.shape
            iw = iw + aw


        self.i_full = nn.Embedding(n, k, _weight=iw)


def loss_mse(X: torch.Tensor):
    """
    MSE loss function for paired predictions.

    This loss function does not require a separate label tensor, because the
    labels are implicit in the structure. :math:`X` has shape (B, 2), where
    column 0 is scores for positive observations and column 1 is scores for
    negative observations.

    Args:
        X(torch.Tensor):
            A tensor of shape (B, 2) storing the prediction scores (in log
            odds).

    Returns:
        torch.Tensor:
            A tensor of shape () with the MSE for the prediction scores.
    """
    # Set up target values of 1/0
    Y = torch.zeros_like(X)
    Y[:, 0] = 1
    Y = Y.to(X.device)

    # Now compute squared error
    sqerr = torch.square(Y - X)

    # And MSE
    n = X.shape[0]
    return sqerr.sum() / (n * 2)


def loss_logistic(X: torch.Tensor):
    """
    Logistic loss function for paired predictions.

    This loss function does not require a separate label tensor, because the
    labels are implicit in the structure. :math:`X` has shape (B, 2), where
    column 0 is scores for positive observations and column 1 is scores for
    negative observations.

    Args:
        X(torch.Tensor):
            A tensor of shape (B, 2) storing the prediction scores (in log
            odds).

    Returns:
        torch.Tensor:
            A tensor of shape () with the negative log likelihood for the
            prediction scores.
    """
    # X is the log odds of 1, but we need column 1 to be the log odds of 0. If
    # we multiply column 0 by 1, and 1 by -1, we will get a new tensor where
    # each element is the log odds of the corresponding rating value, not the
    # always log odds of 1.  A tensor of shape (1, 2) will broadcast with (B, 2)
    # and give us what we need.
    mult = torch.Tensor([1, -1]).reshape((1, 2)).to(X.device)
    Xlo = X * mult

    # Now logsigmoid will convert log odds to log likelihoods
    Xnll = -F.logsigmoid(Xlo)

    # And now we compute the mean negative log likelihood for this batch
    # The total *observations* is n * 2, but since they are always in pairs,
    # dividing by n will suffice to ensure consistent optimization across batches.
    n = X.shape[0]
    return Xnll.sum() / n


def loss_bpr(X: torch.Tensor):
    """
    BPR loss function for paired predictions.

    This loss function does not require a separate label tensor, because the
    labels are implicit in the structure. :math:`X` has shape (B, 2), where
    column 0 is scores for positive observations and column 1 is scores for
    negative observations.

    Args:
        X(torch.Tensor):
            A tensor of shape (B, 2) storing the prediction scores (in log
            odds).

    Returns:
        torch.Tensor:
            A tensor of shape () with the negative log likelihood for the
            prediction scores.
    """
    # For a pair (i, j), we have their scores in columns 0 and 1.
    # The BPR scoring formula is the difference in these scores: i - j
    Xscore = X[:, 0] - X[:, 1]

    # Now logsigmoid will convert that score to a log likelihood
    Xnll = -F.logsigmoid(Xscore)

    # And now we compute the mean of the negative log likelihoods for this batch
    n = X.shape[0]
    return Xnll.sum() / n


class recipeTagMF(Predictor):
    """
    Implementation of a tag-aware hybrid MF in PyTorch.
    """

    _configured_device = None
    _current_device = None

    recipe_data_: recipeData
    _train_data: recipeTrainData
    _model: recipeNet

    def __init__(
        self,
        n_features,
        *,
        batch_size=8192,
        epochs=5,
        reg=0.01,
        components="all",
        loss="bpr",
        device=None,
        rng_spec=None,
    ):
        """
        Initialize the Torch recipe Tag MF predictor.

        This always uses sampled losses — it does not support userwise loss.
        Therefore it omits confidence weights, and gives positive and negative
        examples equal weight.  The ``components`` variable allows the different
        components to be turned on and off; if it is not ``"all"`` or
        ``"none"``, it should be a list of active component names from the
        following list:

        * ``"tags"``
        * ``"genres"``
        * ``"subjects"``

        Args:
            n_features(int):
                The number of latent features (embedding size).
            batch_size(int):
                The batch size for training.  Since this model is relatively
                simple, large batch sizes work well.
            epochs(int):
                The number of training epochs to run.
            reg(float):
                The regularization term to apply to embeddings and biases.
            components(str or list of str):
                The components to activate.
            loss(str):
                The loss function (either ``"bpr"`` or ``"logistic"``).
            rng_spec:
                The random number specification.
        """
        self.n_features = n_features
        self.batch_size = batch_size
        if components == "all":
            self.components = ["tags"]
        elif components == "none":
            self.components = []
        elif isinstance(components, str):
            self.components = [components]
        else:
            self.components = components
        self.epochs = epochs
        self.reg = reg
        self.rng_spec = rng_spec
        self.loss = loss

        self._configured_device = device

    def fit(self, ratings, *, tags, device=None, **kwargs):
        # run the iterations
        timer = util.Stopwatch()

        _log.info("[%s] preparing input data set", timer)
        self._prepare_data(ratings, tags)

        if device is None:
            device = self._configured_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._prepare_model(device)

        # now _data has the training data, and _model has the trainable model

        for epoch in range(self.epochs):
            _log.info("[%s] beginning epoch %d of %d", timer, epoch + 1, self.epochs)

            self._fit_iter()

            unorm = torch.linalg.norm(self._model.u_embed.weight.data).item()
            inorm = torch.linalg.norm(self._model.i_embed.weight.data).item()
            _log.info(
                "[%s] epoch %d finished (|P|=%.3f, |Q|=%.3f)",
                timer,
                epoch + 1,
                unorm,
                inorm,
            )

        _log.info("finished training")
        self._finalize()
        self._cleanup()
        return self

    def _prepare_data(self, ratings, tags):
        "Set up a training data structure for the MF model"
        # index users and items
        _log.info("creating matrix for %d ratings", len(ratings))
        matrix, users, items = sparse_ratings(ratings[["user", "item"]])

        r_users = np.require(matrix.rowinds(), "i4")
        r_items = np.require(matrix.colinds, "i4")

        # set up tag tags
        if "tags" in self.components:
          _log.info("creating tag matrix")
          au_idx = pd.Index(np.unique(tags["tag_id"]))
          tags = tags[["item_id", "tag_id"]]
          tags = tags[tags["item_id"].isin(items)]
          tags = tags.drop_duplicates()
          ba_ino = items.get_indexer(tags["item_id"]).astype("i4")
          ba_ano = au_idx.get_indexer(tags["tag_id"]).astype("i4")
          ba_mat = CSR.from_coo(ba_ino, ba_ano, None, (len(items), len(au_idx)))
        else:
            ba_mat = None



        _log.info("data ready to go")
        recipe_data = recipeData(users, items, ba_mat)
        train_data = recipeTrainData(
            users, items, matrix, r_users, r_items, self.batch_size
        )

        self.recipe_data_ = recipe_data
        self._train_data = train_data

    def _prepare_model(self, train_dev=None):
        self._rng = util.rng(self.rng_spec)
        match self.loss:
            case "mse":
                self._loss = loss_mse
                ub = True
            case "logistic":
                self._loss = loss_logistic
                ub = True
            case "bpr":
                self._loss = loss_bpr
                ub = False
            case _:
                raise ValueError(f"unsupported loss {self.loss}")

        self._model = recipeNet(self.recipe_data_, self.n_features, user_bias=ub)
        if train_dev:
            _log.info("preparing to train on %s", train_dev)
            self._current_device = train_dev
            # move device to model
            self._model = self._model.to(train_dev)
            # set up training features
            self._opt = AdamW(self._model.parameters(), weight_decay=self.reg)

    def _finalize(self):
        "Finalize model training"
        self._model.compact()
        self._model.eval()

    def _cleanup(self):
        "Clean up data not needed after training"
        del self._train_data
        del self._opt, self._loss
        del self._rng

    def to(self, device):
        "Move the model to a different device."
        self._model.to(device)
        self._current_device = device
        return self

    def _fit_iter(self):
        """
        Run one iteration of the recommender training.
        """
        n = self._train_data.n_samples
        # permute the training data
        perm = self._rng.permutation(n)
        loop = tqdm(range(self._train_data.batch_count))

        for i in loop:
            # get the batch - we do this manually, our data is so simple it's faster
            b_start = i * self.batch_size
            b_end = min(b_start + self.batch_size, n)
            # get training rows for this batch
            b_rows = perm[b_start:b_end]
            batch = self._train_data.batch(b_rows)
            batch = batch.to(self._current_device)

            scores = self._model(batch.users, batch.items)
            loss = self._loss(scores)

            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

            loop.set_postfix_str("loss: {:.3f}".format(loss.item()))

        loop.clear()

    def predict_for_user(self, user, items, ratings=None):
        """
        Generate item scores for a user.

        This needs to do two things:

        1. Look up the user's ratings (because ratings is usually none)
        2. Score the items using them

        Note that user and items are both user and item IDs, not positions.
        """

        # convert user and items into rows and columns
        try:
            u_row = self.recipe_data_.users.get_loc(user)
        except KeyError:
            #_log.warn("user %s unknown", user)
            return pd.Series(np.nan, index=items)

        i_cols = self.recipe_data_.items.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]

        u_tensor = torch.IntTensor([u_row])
        i_tensor = torch.from_numpy(i_cols)
        if self._current_device:
            u_tensor = u_tensor.to(self._current_device)
            i_tensor = i_tensor.to(self._current_device)

        # get scores
        with torch.inference_mode():
            scores = self._model(u_tensor, i_tensor).to("cpu")

        # and we can finally put in a series to return
        results = pd.Series(scores, index=scorable)
        return results.reindex(items)  # fill in missing values with nan

    def __str__(self):
        return "Hyrecipe(features={}, reg={})".format(self.n_features, self.reg)

    def __getstate__(self):
        state = dict(self.__dict__)
        if "_model" in state:
            del state["_model"]
            state["_model_weights_"] = self._model.state_dict()
        if "_current_device" in state:
            # we always go back to CPU in pickling
            del state["_current_device"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_model_weights_" in state:
            self._prepare_model()
            self._model.compact(init_only=True)
            self._model.load_state_dict(self._model_weights_)
            # set the model in evaluation mode (not training)
            self._model.eval()
            del self._model_weights_
