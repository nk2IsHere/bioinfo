from typing import NamedTuple, List, Generator, Tuple
from matplotlib import pyplot as plt

from utils import AlignmentScoreMatrix, create_alignment_score_matrix, TracebackAction


class NeedlemanWunschInput(NamedTuple):
    """
    Needleman-Wunsch algorithm input.

    dna1: str - DNA sequence 1.
    dna2: str - DNA sequence 2.
    maximal_final_alignments_count: int - Maximal number of final alignments to generate.
    alignment_score_matrix: AlignmentScoreMatrix - Alignment score matrix.
    gap_penalty: int - Gap penalty.
    """
    dna1: str
    dna2: str
    maximal_final_alignments_count: int = 10
    alignment_score_matrix: AlignmentScoreMatrix = create_alignment_score_matrix()
    gap_penalty: int = -1


class NeedlemanWunschOutput(NamedTuple):
    """
    Needleman-Wunsch algorithm output.

    aligned_dna1: str - Aligned DNA sequence 1.
    aligned_dna2: str - Aligned DNA sequence 2.
    final_score: float - Final alignment score.
    score_matrix: List[List[float]] - Score matrix.
    """
    aligned_dna1: str
    aligned_dna2: str
    final_score: float
    score_matrix: List[List[float]]
    traceback_matrix: List[List[TracebackAction | None]]


def needleman_wunsch_generate_score_matrix(
    dna1: str,
    dna2: str,
    alignment_score_matrix: AlignmentScoreMatrix,
    gap_penalty: int
) -> List[List[float]]:
    """
    Generate Needleman-Wunsch score matrix.

    The score matrix is a 2D list of floats with dimensions (len(dna1) + 1) x (len(dna2) + 1).
    The score matrix is filled with scores for each possible alignment of the two DNA sequences.

    :param dna1: str - DNA sequence 1.
    :param dna2: str - DNA sequence 2.
    :param alignment_score_matrix: AlignmentScoreMatrix - Alignment score matrix.
    :param gap_penalty: int - Gap penalty.
    :return: List[List[float]] - Score matrix.
    """
    score_matrix = [
        [0 for _ in range(len(dna2) + 1)]
        for _ in range(len(dna1) + 1)
    ]

    for i in range(1, len(dna1) + 1):
        score_matrix[i][0] = i * gap_penalty

    for j in range(1, len(dna2) + 1):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, len(dna1) + 1):
        for j in range(1, len(dna2) + 1):
            alignment_score = alignment_score_matrix[dna1[i - 1]][dna2[j - 1]]
            match_mismatch = score_matrix[i - 1][j - 1] + alignment_score
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty

            max_score = max(match_mismatch, delete, insert)
            score_matrix[i][j] = max_score

    return score_matrix


def needleman_wunsch_generate_dna_alignments(
    dna1: str,
    dna2: str,
    traceback: List[TracebackAction]
) -> Tuple[str, str, List[List[TracebackAction | None]]]:
    """
    Generate DNA alignments from Needleman-Wunsch traceback.

    The Needleman-Wunsch traceback is a list of actions that describe the optimal alignment of two DNA sequences.
    The traceback is used to reconstruct the aligned DNA sequences and the traceback matrix.

    :param dna1: str - DNA sequence 1.
    :param dna2: str - DNA sequence 2.
    :param traceback: List[TracebackAction] - Needleman-Wunsch traceback.
    :return: Tuple[str, str, List[List[TracebackAction | None]]] - Aligned DNA sequences and traceback matrix.
    """
    dna1_i = len(dna1)
    dna2_j = len(dna2)
    dna1_aligned = []
    dna2_aligned = []

    traceback_matrix = [
        [None for _ in range(len(dna2) + 1)]
        for _ in range(len(dna1) + 1)
    ]

    for action in traceback:
        traceback_matrix[dna1_i][dna2_j] = action
        match action:
            case TracebackAction.match:
                dna1_aligned.append(dna1[dna1_i - 1])
                dna2_aligned.append(dna2[dna2_j - 1])
                dna1_i -= 1
                dna2_j -= 1
            case TracebackAction.delete:
                dna1_aligned.append(dna1[dna1_i - 1])
                dna2_aligned.append('-')
                dna1_i -= 1
            case TracebackAction.insert:
                dna1_aligned.append('-')
                dna2_aligned.append(dna2[dna2_j - 1])
                dna2_j -= 1

    return (
        ''.join(dna1_aligned),
        ''.join(dna2_aligned),
        traceback_matrix
    )


def needleman_wunsch(needleman_wunsch_input: NeedlemanWunschInput) -> Generator[NeedlemanWunschOutput, None, None]:
    """
    Needleman-Wunsch algorithm.

    The Needleman-Wunsch algorithm is a dynamic programming algorithm that finds the optimal alignment of two DNA
    sequences. The algorithm uses a score matrix to calculate the score of each possible alignment and a traceback
    matrix to reconstruct the optimal alignment.

    :param needleman_wunsch_input: NeedlemanWunschInput - Needleman-Wunsch algorithm input.
    :return: Generator[NeedlemanWunschOutput, None, None] - Generator of Needleman-Wunsch algorithm output.
    """
    dna1, dna2, maximal_final_alignments_count, alignment_score_matrix, gap_penalty = needleman_wunsch_input
    score_matrix = needleman_wunsch_generate_score_matrix(
        dna1,
        dna2,
        alignment_score_matrix,
        gap_penalty
    )

    final_alignments_count = 0
    current_position_with_traceback = [(len(dna1), len(dna2), [])]
    while current_position_with_traceback:
        i, j, current_traceback = current_position_with_traceback.pop()

        if i == 0 and j == 0:
            dna1_aligned, dna2_aligned, traceback_matrix = needleman_wunsch_generate_dna_alignments(
                dna1,
                dna2,
                current_traceback
            )

            yield NeedlemanWunschOutput(
                aligned_dna1=''.join(dna1_aligned[::-1]),
                aligned_dna2=''.join(dna2_aligned[::-1]),
                final_score=float(score_matrix[len(dna1)][len(dna2)]),
                score_matrix=score_matrix,
                traceback_matrix=traceback_matrix
            )

            final_alignments_count += 1
            if final_alignments_count >= maximal_final_alignments_count:
                break

            continue

        # Match
        if i > 0 and j > 0 and score_matrix[i][j] == score_matrix[i - 1][j - 1] + alignment_score_matrix[dna1[i - 1]][dna2[j - 1]]:
            current_position_with_traceback.append((
                i - 1,
                j - 1,
                [*current_traceback, TracebackAction.match]
            ))

        # Delete
        if i > 0 and score_matrix[i][j] == score_matrix[i - 1][j] + gap_penalty:
            current_position_with_traceback.append((
                i - 1,
                j,
                [*current_traceback, TracebackAction.delete]
            ))

        # Insert
        if j > 0 and score_matrix[i][j] == score_matrix[i][j - 1] + gap_penalty:
            current_position_with_traceback.append((
                i,
                j - 1,
                [*current_traceback, TracebackAction.insert]
            ))

        if len(current_position_with_traceback) > maximal_final_alignments_count:
            break


def draw_needleman_wunsch(
    needleman_wunsch_input: NeedlemanWunschInput,
    needleman_wunsch_output: NeedlemanWunschOutput
):
    """
    Draw Needleman-Wunsch algorithm visualization.

    The visualization includes the Needleman-Wunsch score matrix with traceback and final alignment path.

    :param needleman_wunsch_input: NeedlemanWunschInput - Needleman-Wunsch algorithm input.
    :param needleman_wunsch_output: NeedlemanWunschOutput - Needleman-Wunsch algorithm output.
    :return: None
    """
    score_matrix = needleman_wunsch_output.score_matrix

    # Draw score matrix
    # noinspection PyTypeChecker
    plt.matshow(score_matrix, cmap='viridis', origin='upper')
    for i in range(len(needleman_wunsch_input.dna1) + 1):
        for j in range(len(needleman_wunsch_input.dna2) + 1):
            plt.text(j, i, str(float(score_matrix[i][j])), ha='center', va='center', color='black')

    # Draw traceback with overlay
    for i in range(1, len(needleman_wunsch_input.dna1) + 1):
        for j in range(1, len(needleman_wunsch_input.dna2) + 1):
            match needleman_wunsch_output.traceback_matrix[i][j]:
                case TracebackAction.match:
                    plt.text(j - 0.3, i - 0.3, '↖', color='black')
                case TracebackAction.delete:
                    plt.text(j - 0.3, i - 0.3, '↑', color='black')
                case TracebackAction.insert:
                    plt.text(j - 0.3, i - 0.3, '←', color='black')

    # Draw final path with square border
    i, j = len(needleman_wunsch_input.dna1), len(needleman_wunsch_input.dna2)
    while i > 0 or j > 0:
        match needleman_wunsch_output.traceback_matrix[i][j]:
            case TracebackAction.match:
                plt.text(j - 0.3, i - 0.3, '↖', color='red')
                i -= 1
                j -= 1
            case TracebackAction.delete:
                plt.text(j - 0.3, i - 0.3, '↑', color='red')
                i -= 1
            case TracebackAction.insert:
                plt.text(j - 0.3, i - 0.3, '←', color='red')
                j -= 1

    plt.title('Needleman-Wunsch\nScore Matrix')

    plt.xlabel('DNA 2')
    plt.ylabel('DNA 1')
    plt.xticks(range(len(needleman_wunsch_input.dna2) + 1), [''] + list(needleman_wunsch_input.dna2))
    plt.yticks(range(len(needleman_wunsch_input.dna1) + 1), [''] + list(needleman_wunsch_input.dna1))
    plt.show()
