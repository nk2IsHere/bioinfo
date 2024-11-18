from functools import cache

import numpy as np

from itertools import product
from typing import Dict, List

from Bio.Seq import reverse_complement


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
