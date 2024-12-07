from functools import cache

import numpy as np

from itertools import product
from typing import Dict, List, Any

import pandas as pd
from Bio.Seq import reverse_complement
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


def count_kmers(sequence: str, k: int = 4) -> Dict[str, float]:
    """
    A function that counts the frequency of all k-mers in a given sequence

    - Creates a dictionary of all possible k-mers
    - Counts the occurrence of each k-mer in the input sequence by sliding a window of width k with a step of 1
    - Calculates the frequency of each k-mer by dividing its count by the length of the sequence
    (note: do not divide by the number of k-mers in the sequence to ensure normalization across sequences of
    varying lengths)

    Since it is unknown which DNA strand Transcription Factors bind to, reverse complement k-mers should also be counted

    :param sequence: input sequence
    :param k: length of k-mer
    :return: dictionary of k-mers and their frequencies
    """
    sequence_reversed = reverse_complement(sequence)
    kmers = {}

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmers[kmer] = kmers.get(kmer, 0) + 1

    for i in range(len(sequence_reversed) - k + 1):
        kmer = sequence_reversed[i:i + k]
        kmers[kmer] = kmers.get(kmer, 0) + 1

    return kmers


@cache
def all_possible_feature_positions_for_dna(k: int = 4) -> List[str]:
    """
    A function that generates all possible k-mers for DNA sequences

    K-mers and their reverse complements are treated as a single feature
    (e.g., ATTC and GAAT are considered the same 4-mer).

    Example: The dictionary of all 4-mers will contain 136 features. This number represents half of the total 256
    possible combinations (44), plus the number of palindromic k-mers (which are identical to their reverse complements)

    :param k: length of k-mer
    :return: sorted list of all possible k-mers
    """
    all_combinations = set([''.join(i) for i in product('ACGT', repeat=k)])

    kmers = []
    while all_combinations:
        kmer = all_combinations.pop()
        rev_kmer = reverse_complement(kmer)
        if rev_kmer in all_combinations:
            all_combinations.remove(rev_kmer)
        kmers.append(kmer)

    return sorted(kmers)


def transform_kmers_dict_to_feature_vector(kmers: Dict[str, float], k: int = 4) -> np.ndarray:
    """
    A function that transforms a dictionary of k-mers and their frequencies into a feature vector

    - Initializes a feature vector with all zeros
    - Updates the feature vector with the frequency of each k-mer in the input dictionary

    :param kmers: dictionary of k-mers and their frequencies
    :param k: length of k-mer
    :return: feature vector
    """
    all_kmers = all_possible_feature_positions_for_dna(k)
    feature_vector = np.zeros(len(all_kmers))
    for i, kmer in enumerate(all_kmers):
        feature_vector[i] = kmers.get(kmer, 0) + kmers.get(reverse_complement(kmer), 0)

    return feature_vector



def transform_vista_dataset_to_features(
    sequences: pd.DataFrame,
    k: int = 4,
) -> (np.ndarray, np.ndarray):
    """
    Transform the VISTA dataset to feature vectors.
    :param sequences: The VISTA dataset.
    :param k: The length of the k-mer.
    :return: The transformed feature vectors and labels.
    """
    tqdm._instances.clear()
    bar = tqdm(total=len(sequences))
    transformed_X = []
    transformed_y = []

    for i, row in sequences.iterrows():
        bar.update(1)
        sequence = row['seq_hg38']
        kmers = count_kmers(sequence, k)
        transformed_X.append(transform_kmers_dict_to_feature_vector(kmers, k))
        transformed_y.append(row['curation_status'] == 'positive' and 1 or 0)

    transformed_X = np.array(transformed_X)
    transformed_y = np.array(transformed_y)
    bar.close()

    return transformed_X, transformed_y


def select_random_sequence(
    genome: Dict[str, str],
    length: int,
    positive_sequences: pd.DataFrame,
) -> (str, int, int, str):
    """
    Select a random sequence from the genome that does not overlap with the positive sequences.
    :param genome: The genome.
    :param length: The length of the random sequence.
    :param positive_sequences: The positive sequences from the VISTA dataset.
    :return: The chromosome, start, end, and sequence of the random sequence.
    """
    # Extract the start and end positions of the positive sequences
    chr = positive_sequences['coordinate_hg38'].str.split(':').str[0].astype(str)
    start = positive_sequences['coordinate_hg38'].str.split(':').str[1].str.split('-').str[0].astype(int)
    end = positive_sequences['coordinate_hg38'].str.split(':').str[1].str.split('-').str[1].astype(int)

    positive_sequences_positions = sorted([(chr, start, end) for chr, start, end in zip(chr, start, end)])

    @cache
    def compute_allowed_limits_for(chromosome: str) -> List[(int, int)]:
        chromosome_global_begin = 0
        chromosome_global_end = len(str(genome[chromosome]))

        allowed_limits = []
        for i in range(len(positive_sequences_positions)):
            if i == 0:
                allowed_limits.append((chromosome_global_begin, positive_sequences_positions[i][1]))
            else:
                allowed_limits.append((positive_sequences_positions[i-1][2], positive_sequences_positions[i][1]))
        allowed_limits.append((positive_sequences_positions[-1][2], chromosome_global_end))
        return allowed_limits

    # All chromosomes from the positive sequences
    chromosome_choices = positive_sequences['coordinate_hg38'].str.split(':').str[0].unique()

    attempt = 0
    while True:
        chromosome = np.random.choice(chromosome_choices)
        sequence = str(genome[chromosome])

        # Select all limits that are not occupied by the positive sequences
        allowed_limits = compute_allowed_limits_for(chromosome)

        # Select a random position
        start = np.random.randint(0, len(allowed_limits))
        start = allowed_limits[start][0]
        end = start + length
        if end > len(sequence):
            continue

        current_sequence = sequence[start:end]

        if 'N' in current_sequence:
            continue

        attempt += 1
        # print(f'Attempt {attempt}')

        return chromosome, start, end, current_sequence

def transform_vista_dataset_for_classification(
    positive_sequences: pd.DataFrame,
    negative_sequences: pd.DataFrame,
    test_size: int = 400,
    k: int = 4,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Transform the VISTA dataset for classification.
    :param positive_sequences: The positive sequences from the VISTA dataset.
    :param negative_sequences: The negative sequences from the VISTA dataset.
    :param test_size: The size of the test set.
    :param k: The length of the k-mer.
    :return: The transformed feature vectors and labels for the training and test sets.
    """
    train_sequences = pd.concat([positive_sequences.head(-test_size), negative_sequences.head(-test_size)])
    test_sequences = pd.concat([positive_sequences.tail(test_size), negative_sequences.tail(test_size)])

    train_sequences_X, train_sequences_y = transform_vista_dataset_to_features(train_sequences, k)
    test_sequences_X, test_sequences_y = transform_vista_dataset_to_features(test_sequences, k)

    return train_sequences_X, train_sequences_y, test_sequences_X, test_sequences_y

def train_pipeline(train_sequences_X: np.ndarray, train_sequences_y: np.ndarray) -> Pipeline:
    """
    Train a pipeline for classification.
    :param train_sequences_X: The feature vectors of the training set.
    :param train_sequences_y: The labels of the training set.
    :return: The trained pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier())
    ])
    pipeline.fit(train_sequences_X, train_sequences_y)

    return pipeline

def evaluate_pipeline(pipeline: Pipeline, test_sequences_X: np.ndarray, test_sequences_y: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a pipeline for classification.
    :param pipeline: The trained pipeline.
    :param test_sequences_X: The feature vectors of the test set.
    :param test_sequences_y: The labels of the test set.
    :return: The evaluation metrics.
    """
    # Apply stratified k-fold cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(pipeline, test_sequences_X, test_sequences_y, cv=kfold)

    y_pred = pipeline.predict(test_sequences_X)
    report = classification_report(test_sequences_y, y_pred, output_dict=True)

    return {
        'accuracy': results.mean(),
        'accuracy_std': results.std(),
        'classification_report': report,
    }
