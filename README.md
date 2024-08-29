# Recommender Systems Course Projects

This repository contains the coursework for DSCI 641 - Recommender Systems - Winter 2024. The projects focus on building and evaluating various types of recommender systems, utilizing different datasets and methodologies.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Homework 1 - Basic Recommender Systems](#homework-1---basic-recommender-systems)
4. [Homework 2 - Advanced Recommender Systems](#homework-2---advanced-recommender-systems)
5. [Homework 3 - Hybrid Recommender Systems](#homework-3---hybrid-recommender-systems)
6. [Datasets](#datasets)

## Overview

This repository includes three major assignments related to recommender systems:

1. **Homework 1**: Exploration of recommender systems with basic techniques and datasets.
2. **Homework 2**: Implementation and evaluation of advanced recommendation algorithms using LensKit.
3. **Homework 3**: Development of a hybrid recommender system integrating multiple data sources and advanced techniques.

## Project Structure

The repository is structured as follows:

- `HW1/` - Contains the first homework assignment files.
- `HW2/` - Contains the second homework assignment files.
- `HW3/` - Contains the third homework assignment files.
- `Final/` - Contains the Final project files.
- `README.md` - This file.

## Homework 1 - Basic Recommender Systems

**Objective**: In this assignment, we explored two recommender systems datasets: MovieLens 25M and Amazon product ratings. The goal is to compute basic recommendations and analyze the data through statistical methods.

**Key Tasks**:
- Compute basic statistics on the datasets.
- Generate non-personalized recommendations.
- Analyze item similarity using probability and rating-based methods.
- Recommend items for specific users based on item similarity.


## Homework 2 - Advanced Recommender Systems

**Objective**: This assignment focuses on training, evaluating, and exploring several recommendation algorithms using the MovieLens 25M dataset.

**Key Tasks**:
- Partition the data for cross-validation.
- Evaluate multiple collaborative filtering algorithms (e.g., User-User, Item-Item, SVD).
- Implement a content-based recommendation system using the MovieLens Tag Genome.
- Analyze the performance of each algorithm using metrics such as nDCG and RMSE.

## Homework 3 - Hybrid Recommender Systems

**Objective**: The third assignment involves building and evaluating a hybrid recommender system for books, using the UCSD Book Graph dataset. This project integrates user interaction signals with book metadata to generate recommendations.

**Key Tasks**:
- Implement a hybrid recommender system using PyTorch, combining matrix factorization with content-based techniques.
- Evaluate the model using different loss functions (e.g., MSE, BPR).
- Perform hyperparameter tuning to optimize model performance.

## Datasets
    - MovieLens 25M from [here](https://grouplens.org/datasets/movielens/25m/).
    - Amazon ratings data from [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
    - UCSD Book Graph data from [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
