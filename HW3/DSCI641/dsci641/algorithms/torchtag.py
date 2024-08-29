"""
Implementation of a PyTorch-based recommender.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import NamedTuple
from tqdm.auto import tqdm
import math
import numpy as np
import pandas as pd

from csr import CSR
import seedbank

import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.linalg import vecdot

from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util

# I want a logger for information
_log = logging.getLogger(__name__)


class ItemTags(NamedTuple):
    """
    Item tags suitable for input to an EmbeddingBag.
    """

    tag_ids: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_items(cls, matrix: CSR, items: np.ndarray):
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

class MFBatch(NamedTuple):
    "Representation of a single batch."

    "The user IDs (B,1)"
    users: torch.Tensor
    "The item IDs (B,1)"
    items: torch.Tensor
    "The user actions (B,1)"
    actions: torch.Tensor
    "The item authors"
    item_authors: ItemTags
    "The item genres"
    item_genres: ItemTags
    "The item subjects"
    item_subjects: ItemTags


    "The batch size"
    size: int

    def to(self, dev):
        "move this batch to a device"
        return self._replace(
            users=self.users.to(dev),
            items=self.items.to(dev),
            actions=self.actions.to(dev),
            item_authors=self.item_authors.to(dev),
            item_genres=self.item_genres.to(dev),
            item_subjects=self.item_subjects.to(dev)
        )


@dataclass
class SampleEpochData:
    """
    Permuted data for a single epoch of sampled training.
    """

    data: TagMFTrainData
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

    def batch(self, batchno: int) -> MFBatch:
        start = batchno * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        size = end - start

        # find the rows for this sample
        rows = self.permutation[start:end]

        # get user IDs
        uv = self.data.uinds[rows].reshape((size, 1))

        # get item IDs
        iv = np.empty((size, 2), dtype="int32")
        # get positive item IDs
        iv[:, 0] = self.data.matrix.colinds[rows]
        # get negative item IDs
        # it only works with vectors, not matrices, of user ids, so get column
        iv[:, 1], scts = sampling.neg_sample(
            self.data.matrix, uv[:, 0], sampling.sample_unweighted
        )
        # quick debug check
        if np.max(scts) > 8:
            _log.info("%d triples took more than 8 samples", np.sum(scts > 8))

        # get the authors
        item_authors = ItemTags.from_items(self.data.authors_matrix, iv)
        # get the genres
        item_genres = ItemTags.from_items(self.data.genres_matrix, iv)
        # get the subjects
        item_subjects = ItemTags.from_items(self.data.subjects_matrix, iv)

        actions = self.data.actions[rows]

        uv = torch.from_numpy(uv)
        iv = torch.from_numpy(iv)

        # we're done, send to torch and return
        return MFBatch(uv, iv, actions,item_authors, item_genres, item_subjects, size)


@dataclass
class TagMFTrainData:
    """
    Class capturing MF training data/context
    """

    # the user-item matrix
    matrix: CSR
    # the user IDs for each element of the CSR
    uinds: np.ndarray
    # actions
    actions : torch.Tensor
    # the item-authors matrix
    authors_matrix: CSR
    # the item-genres matrix
    genres_matrix: CSR
    # the item-subjects matrix
    subjects_matrix: CSR

    batch_size: int

    @property
    def n_samples(self):
        return self.matrix.nnz

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    def for_epoch(self, rng: np.random.Generator) -> SampleEpochData:
        perm = rng.permutation(self.n_samples)
        return SampleEpochData(self, perm)


class TagMFNet(nn.Module):
    """
    Torch module that defines the matrix factorization model.

    Args:
        n_users(int): the number of users
        n_items(int): the number of items
        n_feats(int): the embedding dimension
    """

    def __init__(self, n_users, n_items, n_authors, n_genres, n_subjects, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.n_users = n_users
        self.n_items = n_items
        self.n_authors = n_authors
        self.n_genres = n_genres
        self.n_subjects = n_subjects

        # user and item bias terms
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)

        # user and item embeddings
        self.u_embed = nn.Embedding(n_users, n_feats)
        self.i_embed = nn.Embedding(n_items, n_feats)

        # tag embeddings - multiple tags per item, so we need EmbeddingBag
        self.a_embed = nn.EmbeddingBag(n_authors, n_feats)
        self.g_embed = nn.EmbeddingBag(n_genres, n_feats)
        self.s_embed = nn.EmbeddingBag(n_subjects, n_feats)

        # rescale all initial values for better starting point
        # they started out as standard normals, those are pretty big
        self.u_bias.weight.data.mul_(0.05)
        self.u_bias.weight.data.square_()
        self.i_bias.weight.data.mul_(0.05)
        self.i_bias.weight.data.square_()

        self.u_embed.weight.data.mul_(0.05)
        self.i_embed.weight.data.mul_(0.05)
        self.a_embed.weight.data.mul_(0.05)
        self.g_embed.weight.data.mul_(0.05)
        self.s_embed.weight.data.mul_(0.05)

    def forward(self, user, item, item_authors, item_genres, item_subjects):
        # look up biases and embeddings
        ub = self.u_bias(user).reshape(user.shape)
        ib = self.i_bias(item).reshape(item.shape)

        uvec = self.u_embed(user)
        _log.debug("uvec shape: %s", uvec.shape)
        ivec = self.i_embed(item)
        _log.debug("ivec shape: %s", ivec.shape)

        # Get author embeddings from the embedding bag
        ia_in, ia_off = item_authors
        avec = self.a_embed(ia_in, ia_off)

        # Get genre embeddings from the embedding bag
        ig_in, ig_off = item_genres
        gvec = self.g_embed(ig_in, ig_off)
        
        # Get subject embeddings from the embedding bag
        is_in, is_off = item_subjects
        svec = self.s_embed(is_in, is_off)


        # embedding bags only support 1D inputs, so we received the
        # item tag data raveled (items stacked atop each other).
        # reshape this so that the items are the right shape
        avec = avec.reshape(ivec.shape)
        gvec = gvec.reshape(ivec.shape)
        svec = svec.reshape(ivec.shape)

        # Sum item and tag vectors to a combined item embedding
        itvec = ivec + avec + gvec + svec

        # compute the inner score
        score = ub + ib + vecdot(uvec, itvec)

        _log.debug(
            "u: %s, i: %s, uv: %s, iv: %s, tv: %s, score: %s",
            ub.shape,
            ib.shape,
            uvec.shape,
            ivec.shape,
            avec.shape,
            gvec.shape,
            svec.shape,
            score.shape,
        )

        # we're done
        return score

# def loss_mse(input: torch.Tensor, target: np.array):
#     # Confidence levels as tensors
#     positive_confidence = torch.full(target.shape, 40)
#     negative_confidence = torch.full(target.shape, 1)
#     nactions = torch.tensor(target.astype(np.int64), dtype=torch.long)

#     # True scores based on interactions
#     true_scores = torch.where(nactions > 0, positive_confidence, negative_confidence)
#     print("Input tensor shape:", input.shape)
#     print("True scores tensor shape:", true_scores.shape)
#     mse_loss = nn.MSELoss()
#     # Now logsigmoid will convert that score to a log likelihood
#     return mse_loss(input, true_scores)

def loss_mse(input: torch.Tensor, nactions: torch.Tensor):
    """
    Mean squared error loss function for implicit feedback, incorporating
    the number of actions (nactions) to reflect interaction intensity.

    Args:
        input (torch.Tensor): A tensor of shape (B, 2) containing prediction scores
                              for positive and sampled negative observations.
        nactions (torch.Tensor): A tensor of shape (B,) containing the number of
                                 actions (interactions) for each positive observation.

    Returns:
        torch.Tensor: A scalar tensor with the mean squared error for the batch.
    """
    # Assuming 'nactions' corresponds to positive observations and you have
    # already sampled an equal number of negative observations with no actions.
    # Define confidence levels based on nactions for positive observations
    positive_confidence = nactions.float()  # Convert nactions to float for MSE calculation
    negative_confidence = torch.full_like(positive_confidence, 1.0)  # Low confidence for negatives

    # Concatenate to create a target tensor matching the input shape
    target = torch.stack((positive_confidence, negative_confidence), dim=1)
    
    # Calculate mean squared error
    mse = F.mse_loss(input, target, reduction='mean')
    
    return mse

def loss_logistic(input: torch.Tensor,**kwargs):
    """
    Logistic loss funcftion for paired predictions.

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
    mult = torch.Tensor([1, -1]).reshape((1, 2)).to(input.device)
    Xlo = input * mult

    # Now logsigmoid will convert log odds to log likelihoods
    Xnll = -F.logsigmoid(Xlo)

    # And now we compute the mean negative log likelihood for this batch
    # The total *observations* is n * 2, but since they are always in pairs,
    # dividing by n will suffice to ensure consistent optimization across batches.
    n = input.shape[0]
    return Xnll.sum() / n

def loss_bpr(input: torch.Tensor, **kwargs):
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
    Xscore = input[:, 0] - input[:, 1]

    # Now logsigmoid will convert that score to a log likelihood
    Xnll = -F.logsigmoid(Xscore)

    # And now we compute the mean of the negative log likelihoods for this batch
    n = input.shape[0]
    return Xnll.sum() / n


class TorchTagMF(Predictor):
    """
    Implementation of a authors, genres, and subject headings aware hybrid MF in PyTorch.
    """

    _configured_device = None
    _current_device = None

    def __init__(
        self,
        n_features,
        *,
        loss = "mse",
        batch_size=8 * 1024,
        lr=0.001,
        epochs=5,
        reg=0.01,
        device=None,
        rng_spec=None,
    ):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            loss(str):
                The loss function to use. Can be ``'mse'`` or ``'logistic'`` or ``'bpr'``.
            batch_size(int):
                The batch size for training.  Since this model is relatively simple,
                large batch sizes work well.
            reg(float):
                The regularization term to apply in AdamW weight decay.
            epochs(int):
                The number of training epochs to run.
            rng_spec:
                The random number specification.
        """
        self.n_features = n_features
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.lr = lr
        self.rng_spec = rng_spec

        self._configured_device = device

    def fit(self, ratings, authors, genres, subjects, device=None, **kwargs):
        """
        Fit the model.  This needs authors, genres, and subjects - call it with::
            algo = TorchMF(...)
            algo = Recommender.adapt(algo)
            algo.fit(ratings, authors = authors, genres = genres, subjects = subjects)
        """
        # run the iterations
        timer = util.Stopwatch()

        _log.info("[%s] preparing input data set", timer)
        self._prepare_data(ratings, authors, genres, subjects)

        if device is None:
            device = self._configured_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._prepare_model(device)
        assert self._model is not None

        # now _data has the training data, and __model has the trainable model

        for epoch in range(self.epochs):
            _log.info("[%s] beginning epoch %d", timer, epoch + 1)

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

    def _prepare_data(self, ratings, authors, genres, subjects):
        "Set up a training data structure for the MF model"
        # index users and items
        rmat, users, items = sparse_ratings(ratings[["user", "item"]])

        self.ratings_ = ratings["nactions"].to_numpy()
        # save the index data for final use
        self.user_index_ = users
        self.item_index_ = items
        
        # create a sparse matrix of item-authors 
        # get item numbers (row numbers)
        authors_inos = items.get_indexer(authors["item"]).astype("i4")
        # filter in case we have authors for an unrated items
        authors = authors[authors_inos >= 0]
        authors_inos = authors_inos[authors_inos >= 0]
        authors_ids = authors['author_id'].astype('category').cat.codes.values.astype("i4")
        authors_vals = (np.ones(len(authors))).astype("f4")
        # make the CSR
        # shape is necessary b/c some items might not have authors
        shape = (len(items), len(authors['author_id'].astype('category').cat.categories))
        authors_mat = CSR.from_coo(authors_inos, authors_ids, authors_vals, shape)

        # create a sparse matrix of item-genres 
        # get item numbers (row numbers)
        genres_inos = items.get_indexer(genres["item"]).astype("i4")
        # filter in case we have genres for an unrated items
        genres = genres[genres_inos >= 0]
        genres_inos = genres_inos[genres_inos >= 0]
        genres_ids = genres['genre_id'].astype('category').cat.codes.values.astype("i4")
        genres_vals = genres["count"].values.astype("f4")
        # make the CSR
        # shape is necessary b/c some items might not have genres
        shape = (len(items), len(genres['genre_id'].astype('category').cat.categories))
        genres_mat = CSR.from_coo(genres_inos, genres_ids, genres_vals, shape)

        
        # create a sparse matrix of item-subjects
        # get item numbers (row numbers)
        subjects_inos = items.get_indexer(subjects["item"]).astype("i4")
        # filter in case we have subjects for an unrated items
        subjects = subjects[subjects_inos >= 0]
        subjects_inos = subjects_inos[subjects_inos >= 0]
        subjects_ids = subjects['subj_id'].astype('category').cat.codes.values.astype("i4")
        subjects_vals = (np.ones(len(subjects))).astype("f4")
        # make the CSR
        # shape is necessary b/c some items might not have authors
        shape = (len(items), len(subjects['subj_id'].astype('category').cat.categories))
        subjects_mat = CSR.from_coo(subjects_inos, subjects_ids, subjects_vals, shape)

        # set up the training data
        self._data = TagMFTrainData(rmat,rmat.rowinds(), torch.tensor(ratings["nactions"].values.astype(np.int64), dtype=torch.long),authors_mat, genres_mat, subjects_mat, self.batch_size)

        self.user_index_ = users
        self.item_index_ = items
        self.item_authors_ = authors_mat
        self.item_genres_ = genres_mat
        self.item_subjects_ = subjects_mat

    def _prepare_model(self, train_dev=None):
        n_users = len(self.user_index_)
        n_items = len(self.item_index_)
        n_authors = self.item_authors_.ncols
        n_genres = self.item_genres_.ncols
        n_subjects = self.item_subjects_.ncols
        self._rng = seedbank.numpy_rng(self.rng_spec)
        model = TagMFNet(n_users, n_items, n_authors, n_genres, n_subjects, self.n_features)
        self._model = model
        if train_dev:
            _log.info("preparing to train on %s", train_dev)
            self._current_device = train_dev
            # move device to model
            self._model = model.to(train_dev)
            # put model in training mode
            self._model.train(True)
            # set up loss function
            match self.loss:
                case "mse":
                    self._loss = loss_mse
                case "logistic":
                    self._loss = loss_logistic
                case "bpr":
                    self._loss = loss_bpr
                case _:
                    raise ValueError(f"invalid loss {self.loss}")
            # set up training features            
            self._opt = AdamW(
                self._model.parameters(), lr=self.lr, weight_decay=self.reg
            )

    def _finalize(self):
        "Finalize model training"
        self._model.eval()

    def _cleanup(self):
        "Clean up data not needed after training"
        del self._data
        del self._loss, self._opt
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
        # permute the training data
        epoch_data = self._data.for_epoch(self._rng)
        loop = tqdm(range(epoch_data.batch_count))
        for i in loop:
            batch = epoch_data.batch(i).to(self._current_device)

            # compute scores and loss
            pred = self._model(batch.users, batch.items, batch.item_authors, batch.item_genres, batch.item_subjects)
    
            loss = self._loss(input = pred, nactions = batch.actions)

            # update model
            loss.backward()
            self._opt.step()
            self._opt.zero_grad()

            loop.set_postfix_str("loss: {:.3f}".format(loss.item()))

            if i % 100 == 99:
                _log.debug("batch %d has NLL %s", i, loss.item())

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
        u_row = self.user_index_.get_loc(user)
        u_tensor = torch.IntTensor([u_row])

        i_cols = self.item_index_.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]
        i_tensor = torch.from_numpy(i_cols)

        authors_info = ItemTags.from_items(self.item_authors_, i_cols)
        genres_info = ItemTags.from_items(self.item_genres_, i_cols)
        subjects_info = ItemTags.from_items(self.item_subjects_, i_cols)
        if self._current_device:
            u_tensor = u_tensor.to(self._current_device)
            i_tensor = i_tensor.to(self._current_device)
            authors_info = authors_info.to(self._current_device)
            genres_info = genres_info.to(self._current_device)
            subjects_info = subjects_info.to(self._current_device)

        # get scores
        with torch.inference_mode():
            scores = self._model(u_tensor, i_tensor, authors_info, genres_info, subjects_info).to("cpu")

        # and we can finally put in a series to return
        results = pd.Series(scores, index=scorable)
        return results.reindex(items)  # fill in missing values with nan

    def __str__(self):
        return "TorchTagMF(features={}, loss={}, reg={})".format(self.n_features, self.loss, self.reg)

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
            self._model.load_state_dict(self._model_weights_)  # type: ignore
            # put model in evaluation mode
            self._model.eval()
            del self._model_weights_  # type: ignore
