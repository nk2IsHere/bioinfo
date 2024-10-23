from enum import IntEnum
from typing import Literal, Dict

import pandas as pd

Allele = Literal['A', 'C', 'G', 'T']


class AlignmentScoreMatrix(Dict[Allele, Dict[Allele, int]]):
    """
    Alignment score matrix.

    The alignment score matrix is a dictionary of dictionaries that maps a pair of alleles to an alignment score.

    Example:
    {
        'A': {'A': 1, 'C': -1, 'G': -1, 'T': -1},
        'C': {'A': -1, 'C': 1, 'G': -1, 'T': -1},
        'G': {'A': -1, 'C': -1, 'G': 1, 'T': -1},
        'T': {'A': -1, 'C': -1, 'G': -1, 'T': 1}
    }
    """

    def __getitem__(self, key: Allele) -> Dict[Allele, int]:
        return super().__getitem__(key)

    def __setitem__(self, key: Allele, value: Dict[Allele, int]) -> None:
        super().__setitem__(key, value)

    def __str__(self) -> str:
        return str({key: dict(value) for key, value in self.items()})

    @staticmethod
    def from_dict(d: Dict[Allele, Dict[Allele, int]]) -> 'AlignmentScoreMatrix':
        """
        Create an alignment score matrix from a dictionary.

        The dictionary is expected to have the following structure:
        {
            'A': {'A': 1, 'C': -1, 'G': -1, 'T': -1},
            'C': {'A': -1, 'C': 1, 'G': -1, 'T': -1},
            'G': {'A': -1, 'C': -1, 'G': 1, 'T': -1},
            'T': {'A': -1, 'C': -1, 'G': -1, 'T': 1}
        }

        :param d: Dict[Allele, Dict[Allele, int]] - Dictionary with alignment scores.
        :return: AlignmentScoreMatrix - Alignment score matrix.
        """
        return AlignmentScoreMatrix(d)

    @staticmethod
    def from_df(df: pd.DataFrame) -> 'AlignmentScoreMatrix':
        """
        Create an alignment score matrix from a DataFrame.

        The DataFrame is expected to have the following structure:
        A,G,C,T
        5,-4,-4,-1
        -4,5,-4,-1
        -4,-4,5,-1
        -1,-1,-1,5

        :param df: pd.DataFrame - DataFrame with alignment scores.
        :return: AlignmentScoreMatrix - Alignment score matrix.
        """
        df = df.to_dict()

        # Convert the DataFrame to a dictionary of dictionaries
        # Keys have to be mapped from positions to alleles that are defined by csv header
        allele_positions = list(df.keys())
        for allele in allele_positions:
            df[allele] = {allele_positions[i]: df[allele][i] for i in range(len(allele_positions))}

        return AlignmentScoreMatrix.from_dict(df)


def create_alignment_score_matrix(
    match_score: int = 1,
    mismatch_score: int = -1
) -> AlignmentScoreMatrix:
    """
    Create a default alignment score matrix.

    :param match_score: int - Match score.
    :param mismatch_score: int - Mismatch score.
    :return: AlignmentScoreMatrix - Alignment score matrix.
    """
    return AlignmentScoreMatrix.from_dict({
        'A': {'A': match_score, 'C': mismatch_score, 'G': mismatch_score, 'T': mismatch_score},
        'C': {'A': mismatch_score, 'C': match_score, 'G': mismatch_score, 'T': mismatch_score},
        'G': {'A': mismatch_score, 'C': mismatch_score, 'G': match_score, 'T': mismatch_score},
        'T': {'A': mismatch_score, 'C': mismatch_score, 'G': mismatch_score, 'T': match_score}
    })


class TracebackAction(IntEnum):
    match = 0
    delete = 1
    insert = 2
