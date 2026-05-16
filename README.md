# Session-Based Movie Recommendation Engine

A deep learning recommendation system built with TensorFlow/Keras, MongoDB Atlas, and FastAPI.
The model learns user preferences from behavioral data and recommends movies in real time using a Two-Tower neural architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)
   - [NoSQL Schema Design](#nosql-schema-design)
   - [Data Pipeline](#data-pipeline)
   - [Two-Tower Model](#two-tower-model)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Inference API](#inference-api)
7. [Results](#results)
8. [API Reference](#api-reference)
9. [Reproducing the Project](#reproducing-the-project)
10. [Engineering Notes](#engineering-notes)
11. [Next Steps](#next-steps)

---

## Overview

Recommendation systems are the core of platforms like Netflix and Amazon.
This project implements a **session-based recommendation engine** that adapts to a user's recent interaction history rather than relying on static preferences.

The key design decision is to keep behavioral data, such as click events, ratings, session history, in MongoDB Atlas, whose flexible document schema handles nested JSON naturally.
The Two-Tower model then transforms this data into dense vector representations, where recommendation becomes a nearest-neighbor search in embedding space.

---

## Architecture

```
MongoDB Atlas
  users       movies       ratings      sessions
     |            |            |             |
     +------------+------------+-------------+
                        |
                  Data Pipeline
             (encoders, negative sampling,
              tf.data.Dataset)
                        |
              Two-Tower Model (Keras)
             /                        \
      User Tower                  Item Tower
   Embedding(6040, 64)         Embedding(3706, 64)
   Dense(128, relu)            Dense(128, relu)
   Dense(64)                   Dense(64)
   L2 normalize                L2 normalize
             \                        /
              Dot product (cosine similarity)
                        |
                   BPR Loss
                        |
              Trained Embeddings
                        |
                  FastAPI
         /recommend/{user_id}
         /similar/{movie_id}
         /movie/{movie_id}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Database | MongoDB Atlas M0 |
| Deep learning | TensorFlow 2.20 + Keras |
| Data processing | Pandas, NumPy |
| API | FastAPI + Uvicorn |
| Tunnel (dev) | ngrok |
| Environment | Google Colab |
| Dataset | MovieLens 1M |

---

## Dataset

**MovieLens 1M** — published by GroupLens Research, University of Minnesota.

- 1,000,209 ratings
- 6,040 users
- 3,706 movies
- Ratings from 1 to 5, with timestamps
- Movie metadata: title, genres (multi-label)
- User metadata: gender, age group, occupation, zip code

Source: https://grouplens.org/datasets/movielens/1m/

---

## Project Structure

```
recsys-two-tower/
|
|-- artifacts/
|   |-- item_embeddings.npy         # (3706, 64) item vectors
|   |-- user_embeddings.npy         # (6040, 64) user vectors
|   |-- user2idx.pkl                # user_id -> embedding index
|   |-- movie2idx.pkl               # movie_id -> embedding index
|   |-- idx2movie.pkl               # embedding index -> movie_id
|   |-- movies_lookup.pkl           # movie_id -> {title, genres}
|   |-- model_weights.weights.h5    # trained model weights
|
|-- notebook/
|   |-- recsys_two_tower.ipynb      # full notebook, all phases and FastAPI application
|
|-- README.md
```

---

## How It Works

### NoSQL Schema Design

Four collections are used in MongoDB Atlas.

**users** and **movies** store raw attributes.
The `genres` field on movies is stored as an array rather than a string, enabling direct array queries without parsing.

**ratings** stores every interaction as a flat document with `user_id`, `movie_id`, `rating`, and `timestamp`.
A compound index on `(user_id, timestamp DESC)` makes per-user history retrieval fast.

**sessions** is the most important collection for the model.
Each document represents one user's recent interaction history, with movie metadata embedded directly — no join required at training time:

```json
{
  "user_id": 1,
  "gender": "F",
  "age": 1,
  "occupation": 10,
  "interactions": [
    {
      "movie_id": 1193,
      "title": "One Flew Over the Cuckoo's Nest (1975)",
      "genres": ["Drama"],
      "rating": 5,
      "timestamp": 978300760
    }
  ]
}
```

This denormalized schema trades storage for query speed, a deliberate choice for a system where reads vastly outnumber writes.

### Data Pipeline

Three steps prepare the data for training.

**Encoding:** User IDs and movie IDs are mapped to contiguous integer indices starting at zero.
Keras `Embedding` layers require this, the original IDs have gaps.

**Negative sampling:** For each positive interaction, decided by 'rating >= 4', one negative item is sampled.
Negatives are drawn proportionally to item popularity rather than uniformly at random.
Popular items are harder negatives, the model must learn to distinguish a user's actual preference from a globally popular item they simply have not seen yet.

**tf.data.Dataset:** Triplets `(user_idx, pos_movie_idx, neg_movie_idx)` are wrapped in a `tf.data.Dataset` with shuffling, batching, and prefetching.
Training set: 517,753 samples. Validation set: 57,528 samples.

### Two-Tower Model

The model consists of two independent neural networks, one for users, one for items, that produce vectors in the same 64-dimensional space.

```
User tower:
  Input: user_idx (integer)
  Embedding(NUM_USERS, 64)
  Dense(128, activation='relu')
  Dropout(0.2)
  Dense(64)
  L2 normalize

Item tower:
  Input: movie_idx (integer)
  Embedding(NUM_MOVIES, 64)
  Dense(128, activation='relu')
  Dropout(0.2)
  Dense(64)
  L2 normalize
```

L2 normalization constrains all vectors to the unit hypersphere.
The dot product of two normalized vectors equals their cosine similarity, which lies in [-1, 1].
This prevents the model from learning to assign high scores simply by making some embeddings large in magnitude.

At inference time, item embeddings are pre-computed once.
Recommendation for a user requires only one forward pass through the user tower, followed by a dot product against the full item matrix — an O(N) operation that can be replaced with approximate nearest-neighbor search for very large catalogs.

### Training

Loss function: **Bayesian Personalised Ranking (BPR)**

```
loss = -mean( log( sigmoid( score(u, pos) - score(u, neg) ) ) )
```

BPR directly optimizes the relative ordering of positive over negative items, which is the correct objective for a ranking system.
Mean Squared Error or cross-entropy on ratings would optimize for score accuracy, not ranking quality.

Optimizer: Adam, initial learning rate 1e-3.
Callbacks: EarlyStopping (patience=5) and ReduceLROnPlateau (patience=3, factor=0.5).
The model converged in approximately 15 epochs on a Colab T4 GPU.

### Evaluation

Evaluation follows the standard protocol used in academic recommendation papers:

For each user in the evaluation set:
1. Hold out the most recent positive interaction as ground truth
2. Sample 99 items the user has never interacted with as negatives
3. Score all 100 candidates with the model
4. Measure whether the ground truth appears in the top K

This protocol is more interpretable than full-catalog ranking and allows fair comparison with published baselines.

### Inference API

The FastAPI application loads the pre-computed embeddings and model artifacts at startup.
On each request to `/recommend/{user_id}`, it:

1. Looks up the user's embedding vector
2. Computes dot products against all item embeddings
3. Sets scores of already-watched items to -inf
4. Returns the top-K items with titles and genres
5. Attaches the user's recent history from MongoDB for context

---

## Results

| Metric | Value |
|---|---|
| Recall@1 | 10.0% |
| Recall@5 | 36.4% |
| Recall@10 | 52.1% |
| Recall@20 | 67.2% |
| Recall@50 | 89.2% |
| NDCG@10 | 0.29 |
| NDCG@50 | 0.36 |

Evaluation protocol: 1 positive + 99 random negatives, 500 users sampled.
Random baseline Recall@10: 10.0%.
The model achieves 5x improvement over random at K=10.

**Qualitative check — User 1** - watches animated family films: Pocahontas, Mulan, Antz:

| Rank | Title | Genres |
|---|---|---|
| 1 | Parenthood (1989) | Comedy, Drama |
| 2 | Iron Giant, The (1999) | Animation, Children's |
| 3 | Field of Dreams (1989) | Drama |
| 6 | Grand Day Out, A (1992) | Animation, Comedy |
| 10 | Chicken Run (2000) | Animation, Children's, Comedy |

The model correctly maps this user to the family/feel-good region of the embedding space.

---

## API Reference

Base URL: `https://<ngrok-subdomain>.ngrok-free.app`

### GET /recommend/{user_id}

Returns top-K movie recommendations for a user.

Query parameters:
- `top_k` (integer, default 10) - number of recommendations to return

Response:
```json
{
  "user_id": 1,
  "top_k": 10,
  "recommendations": [
    {
      "rank": 1,
      "score": 0.9744,
      "movie_id": 3526,
      "title": "Parenthood (1989)",
      "genres": ["Comedy", "Drama"]
    }
  ],
  "recent_history": [
    {
      "title": "Pocahontas (1995)",
      "rating": 5,
      "genres": ["Animation", "Children's", "Musical", "Romance"]
    }
  ]
}
```

### GET /similar/{movie_id}

Returns movies most similar to the given movie by embedding cosine similarity.

Query parameters:
- `top_k` (integer, default 10)

### GET /movie/{movie_id}

Returns metadata for a single movie.

### GET /docs

Interactive Swagger UI with all endpoints.

---

## Reproducing the Project

**Requirements:**
- Google Colab account (free T4 GPU)
- MongoDB Atlas account (free M0 cluster)
- ngrok account (free authtoken)

**Steps:**

1. Clone this repository and open `notebook/recsys_two_tower.ipynb` in Google Colab
2. In Colab, go to Secrets and add:
   - `MONGO_URI` — your MongoDB Atlas connection string
   - `NGROK_TOKEN` — your ngrok authtoken
3. In MongoDB Atlas, set Network Access to allow connections from anywhere (0.0.0.0/0) for development
4. Run all cells in order
5. The data ingestion block only needs to run once - data persists in Atlas across sessions
6. For subsequent sessions, skip to the verification block to confirm data is present before continuing

**Estimated runtimes:**
- Data ingestion: ~20 minutes
- Session building: ~5 minutes
- Model training: ~8 minutes
- API startup: ~30 seconds

---

## Engineering Notes

These are issues encountered during development and how they were resolved.

**MongoDB Atlas SSL handshake failure on Colab**

The Python SSL stack on Colab 2024+ has a TLS version mismatch with Atlas.
Fixed by passing `tls=True, tlsAllowInvalidCertificates=True` to `MongoClient`.

**np.int64 not JSON serializable**

NumPy integer types are not handled by Python's built-in `json` module.
Fixed with a custom `JSONEncoder` subclass that maps `np.integer` to `int` and `np.floating` to `float` before serialization.

**Negative sampling quality**

Initial uniform random negative sampling produced a model that recommended only globally popular items regardless of user identity.
The fix was popularity-weighted negative sampling by drawing negatives proportionally to item frequency makes the task harder and forces the model to learn user-specific signal rather than global popularity.

**Embedding magnitude saturation**

Without L2 normalization, all recommendation scores clustered near 0.999.
The dot product was measuring embedding magnitude rather than directional alignment.
Adding a normalization layer at the output of each tower constrains vectors to the unit sphere and converts dot products to cosine similarity.

---

## Next Steps

The following improvements would be natural extensions of this project.

**Model**
- Add side features to the towers: user age, occupation, movie release year
- Implement a re-ranking layer that takes the top 50 candidates from the Two-Tower and applies a more expensive cross-attention model
- Experiment with transformer-based session encoders to replace the simple embedding lookup for users

**Infrastructure**
- Replace the full dot product scan with approximate nearest neighbor search for sub-millisecond latency at scale
- Add a background job that rebuilds session documents in MongoDB as new ratings arrive
- Containerize the API with Docker and deploy to a persistent host

**Evaluation**
- Add A/B testing simulation: compare Two-Tower against a popularity baseline and a matrix factorization baseline on the same evaluation set
- Compute coverage and diversity metrics alongside Recall and NDCG
