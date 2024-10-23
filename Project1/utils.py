from enum import IntEnum
from typing import Literal, Dict

import pandas as pd

Allele = Literal['A', 'C', 'G', 'T']


class AlignmentScoreMatrix(Dict[Allele, Dict[Allele, int]]):

    def __getitem__(self, key: Allele) -> Dict[Allele, int]:
        return super().__getitem__(key)

    def __setitem__(self, key: Allele, value: Dict[Allele, int]) -> None:
        super().__setitem__(key, value)

    def __str__(self) -> str:
        return str({key: dict(value) for key, value in self.items()})

    @staticmethod
    def from_dict(d: Dict[Allele, Dict[Allele, int]]) -> 'AlignmentScoreMatrix':
        return AlignmentScoreMatrix(d)

    @staticmethod
    def from_df(df: pd.DataFrame) -> 'AlignmentScoreMatrix':
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
