# Information Diffusion Cascade Prediction

This repository implements a complete pipeline for predicting the growth of information cascades.  The implementation is inspired by the paper **"Can Cascades be Predicted?"** and is adapted to work on a simple Weibo-style dataset where each line describes a cascade with a list of retweet paths and their timestamps.

The goal of the project is to build a binary classifier which, given the first *k* retweets of a cascade, predicts whether the cascade will at least double in size (i.e. reach at least 2*k* retweets).  The pipeline is designed to be modular and easy to understand so that you can extend it or substitute different feature sets and models.

## Repository Structure

```text
cascade-prediction-networks/
├── data/
    └── sample_weibo.txt # small example dataset (for sanity checks)
├── src/
    └── cascade.py # simple container class for cascades
    └── parse_data.py # parser for the Weibo dataset
    └── build_tree.py # utilities for building cascade trees and computing depths
    └── prefix.py # functions for generating k‑prefix cascades
    └── features.py # temporal and structural feature extraction
    └── labels.py # construction of binary labels using the 2k rule
    └── models.py # training and prediction routines
    └── evaluation.py # metrics and evaluation utilities
    └── main.py # orchestrates the full pipeline from raw data to results
├── requirements.txt # Python dependencies
├── README.md # this document
```

Each module is independent but designed to work together.  You can run the entire pipeline using the **main** script.

## Installation

1. **Clone or copy** this repository into your working directory.
2. Create a virtual environment (recommended) and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The primary dependencies are pandas, networkx, numpy and scikit‑learn.

## Dataset Format

The code expects a tab‑separated input file where each line describes one cascade. The columns are:

1. message_id – unique identifier for the cascade

1. root_user – user ID of the original poster

1. publish_time – Unix timestamp of the original post

1. retweet_number – number of retweets recorded

1. retweet paths – a space‑separated list of retweet paths and their timestamps

Each retweet path is of the form u1/u2/…/un:Δt where u1 is the root, un is the user who retweeted, and Δt is the elapsed time in seconds since the publish time. Intermediate users reflect the path through which the message diffused (e.g. 77/86/87:40637 indicates that user 87 retweeted from user 86 who previously retweeted from user 77, at 40 637 seconds after the original post). If the same user appears multiple times in different paths, only the earliest retweet is kept.

A small example file (data/sample_weibo.txt) is provided for testing. You should replace this with your full dataset when running the experiments.

## Running the Pipeline
The main script orchestrates the pipeline and can be run from the project root as follows:

```bash
python src/main.py --input data/sample_weibo.txt --ks 5 10 --test_size 0.3 --random_state 42
```

Arguments:

- input – path to the dataset file.

- ks – one or more values of k, the number of initial retweets used to form the prefix. You can specify multiple values and the script will train separate models for each.

- test_size – fraction of cascades to hold out for evaluation (default is 0.3).

- random_state – random seed for reproducibility.

The script will:

1. Parse the dataset and build full cascade objects.

1. Generate k‑prefix cascades for each specified k.

1. Extract temporal and structural features from each prefix.

1. Construct binary labels using the doubling rule (1 if the cascade will at least double in size, otherwise 0).

1. Train a logistic regression classifier (and a random forest as a bonus) on 70% of the data and evaluate on the remaining 30%.

1. Print the evaluation metrics (accuracy, F1 score and AUC) and show the most important features for the random forest.

The results are printed to the console and can be saved to files for inclusion in your report.

## Expected Outputs
When you run the pipeline on the example file with k=5 and k=10 you should expect to see output similar to the following (exact numbers will differ because the sample is tiny):

```yaml
=== Results for k=5 ===
Number of prefix cascades: 3
Label distribution: Counter({1: 2, 0: 1})
Logistic Regression accuracy: 0.66
Random Forest accuracy: 1.00
Random Forest top features:
  time_to_k           0.35
  mean_inter_time     0.22
  depth               0.18
  leaves              0.14
  avg_depth           0.11

=== Results for k=10 ===
...
```

The small sample dataset contains only a few cascades, so the scores are not meaningful; however, they confirm that the pipeline runs end‑to‑end. For a larger dataset you should obtain more stable metrics (typically accuracy around 0.6–0.7 for logistic regression on Weibo data with k=5 or k=10).

## Extending and Customising
- More features – you can add user metadata (e.g. follower counts), content features or exposure information if available. Modify features.py to compute additional metrics.

- Different thresholds – instead of the doubling rule you can use other size thresholds (e.g. 3k). Change labels.construct_labels accordingly.

- Other models – feel free to experiment with other classifiers (SVMs, gradient boosting, neural networks) by editing models.py.

- Cross‑validation – for more robust evaluation use k‑fold cross‑validation instead of a single train/test split.

## License
This code is provided for educational purposes as part of a university assignment. Feel free to adapt it for your own experiments.
