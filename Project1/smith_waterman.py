from typing import NamedTuple, List, Tuple

from utils import AlignmentScoreMatrix, create_alignment_score_matrix, TracebackAction


class SmithWatermanInput(NamedTuple):
    """
    Smith-Waterman algorithm input.

    dna1: str - DNA sequence 1.
    dna2: str - DNA sequence 2.
    alignment_score_matrix: AlignmentScoreMatrix - Alignment score matrix.
    gap_penalty: int - Gap penalty.
    """
    dna1: str
    dna2: str
    alignment_score_matrix: AlignmentScoreMatrix = create_alignment_score_matrix()
    gap_penalty: int = -1


class SmithWatermanOutput(NamedTuple):
    """
    Smith-Waterman algorithm output.

    aligned_dna1: str - Aligned DNA sequence 1.
    aligned_dna2: str - Aligned DNA sequence 2.
    final_score: float - Final alignment score.
    score_matrix: List[List[float]] - Score matrix.
    traceback_matrix: List[List[TracebackAction | None]] - Traceback matrix.
    """
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
    """
    Generate Smith-Waterman score and traceback matrices.
    
    The score matrix is a 2D list of floats with dimensions (len(dna1) + 1) x (len(dna2) + 1).
    The score matrix is filled with scores for each possible alignment of the two DNA sequences.
    
    The traceback matrix is a 2D list of TracebackAction or None with dimensions (len(dna1) + 1) x (len(dna2) + 1).
    The traceback matrix is filled with actions that lead to the optimal alignment.
    
    The final score is the highest score in the score matrix.
    The final i and j indices are the indices of the final score in the score matrix.
    
    :param dna1: str - DNA sequence 1.
    :param dna2: str - DNA sequence 2.
    :param alignment_score_matrix: AlignmentScoreMatrix - Alignment score matrix.
    :param gap_penalty: int - Gap penalty.
    :return: Tuple[List[List[float]], List[List[TracebackAction | None]], float, int, int] - Score matrix, traceback matrix, final score, final i index, final j index.
    """
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
    """
    Smith-Waterman algorithm.

    The Smith-Waterman algorithm is a dynamic programming algorithm that performs local sequence alignment.
    The algorithm finds the optimal local alignment of two DNA sequences.

    :param input: SmithWatermanInput - Smith-Waterman algorithm input.
    :return: SmithWatermanOutput - Smith-Waterman algorithm output.
    """
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
