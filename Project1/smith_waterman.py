from typing import NamedTuple, List, Tuple

from utils import AlignmentScoreMatrix, create_alignment_score_matrix, TracebackAction


class SmithWatermanInput(NamedTuple):
    dna1: str
    dna2: str
    alignment_score_matrix: AlignmentScoreMatrix = create_alignment_score_matrix()
    gap_penalty: int = -1


class SmithWatermanOutput(NamedTuple):
    aligned_dna1: str
    aligned_dna2: str
    final_score: float
    score_matrix: List[List[float]]
    traceback_matrix: List[List[TracebackAction | None]]


def smith_waterman_generate_matrices(
    dna1: str,
    dna2: str,
    alignment_score_matrix:
    AlignmentScoreMatrix,
    gap_penalty: int
) -> Tuple[
    List[List[float]],
    List[List[TracebackAction | None]],
    float,
    int,
    int
]:
    n = len(dna1)
    m = len(dna2)

    score_matrix = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    traceback_matrix = [[None for _ in range(m + 1)] for _ in range(n + 1)]

    max_score = 0
    max_i = 0
    max_j = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = score_matrix[i - 1][j - 1] + alignment_score_matrix[dna1[i - 1]][dna2[j - 1]]
            delete_score = score_matrix[i - 1][j] + gap_penalty
            insert_score = score_matrix[i][j - 1] + gap_penalty

            score = max(0, match_score, delete_score, insert_score)
            score_matrix[i][j] = score

            if score == 0:
                traceback_matrix[i][j] = None
            elif score == match_score:
                traceback_matrix[i][j] = TracebackAction.match
            elif score == delete_score:
                traceback_matrix[i][j] = TracebackAction.delete
            elif score == insert_score:
                traceback_matrix[i][j] = TracebackAction.insert

            if score > max_score:
                max_score = score
                max_i = i
                max_j = j

    return score_matrix, traceback_matrix, max_score, max_i, max_j


def smith_waterman(input: SmithWatermanInput) -> SmithWatermanOutput:
    dna1, dna2, alignment_score_matrix, gap_penalty = input
    score_matrix, traceback_matrix, max_score, max_i, max_j = smith_waterman_generate_matrices(
        dna1,
        dna2,
        alignment_score_matrix,
        gap_penalty
    )

    aligned_dna1 = ''
    aligned_dna2 = ''
    i = max_i
    j = max_j
    while i > 0 and j > 0:
        if traceback_matrix[i][j] == TracebackAction.match:
            aligned_dna1 = dna1[i - 1] + aligned_dna1
            aligned_dna2 = dna2[j - 1] + aligned_dna2
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == TracebackAction.delete:
            aligned_dna1 = dna1[i - 1] + aligned_dna1
            aligned_dna2 = '-' + aligned_dna2
            i -= 1
        elif traceback_matrix[i][j] == TracebackAction.insert:
            aligned_dna1 = '-' + aligned_dna1
            aligned_dna2 = dna2[j - 1] + aligned_dna2
            j -= 1
        else:
            break

    return SmithWatermanOutput(
        aligned_dna1,
        aligned_dna2,
        max_score,
        score_matrix,
        traceback_matrix
    )
