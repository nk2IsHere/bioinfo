import unittest
from needleman_wunsch import needleman_wunsch, NeedlemanWunschInput
from smith_waterman import smith_waterman, SmithWatermanInput
from utils import create_alignment_score_matrix


class TestSequenceAlignments(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.match_score = 1
        self.mismatch_score = -1
        self.gap_penalty = -1
        self.alignment_score_matrix = create_alignment_score_matrix(
            match_score=self.match_score,
            mismatch_score=self.mismatch_score
        )

    def test_needleman_wunsch_identical_sequences(self):
        """Test NW alignment of identical sequences."""
        dna1 = "AGCT"
        dna2 = "AGCT"

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(result.aligned_dna1, "AGCT")
        self.assertEqual(result.aligned_dna2, "AGCT")
        self.assertEqual(result.final_score, 4.0)  # 4 matches * match_score

    def test_needleman_wunsch_with_gaps(self):
        """Test NW alignment requiring gaps."""
        dna1 = "AGTC"
        dna2 = "AGC"

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        # Expected alignment:
        # AGTC
        # AG-C
        self.assertEqual(result.aligned_dna1, "AGTC")
        self.assertEqual(result.aligned_dna2, "AG-C")
        self.assertEqual(result.final_score, 2.0)  # 3 matches * match_score + 1 gap * gap_penalty

    def test_needleman_wunsch_with_mismatches(self):
        """Test NW alignment with mismatches."""
        dna1 = "AGCT"
        dna2 = "AGTT"

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(result.aligned_dna1, "AGCT")
        self.assertEqual(result.aligned_dna2, "AGTT")
        self.assertEqual(result.final_score, 2.0)  # 3 matches * match_score + 1 mismatch * mismatch_score

    def test_needleman_wunsch_empty_sequence(self):
        """Test NW alignment with an empty sequence."""
        dna1 = "AGC"
        dna2 = ""

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(result.aligned_dna1, "AGC")
        self.assertEqual(result.aligned_dna2, "---")
        self.assertEqual(result.final_score, -3.0)  # 3 gaps * gap_penalty

    def test_smith_waterman_identical_sequences(self):
        """Test SW alignment of identical sequences."""
        dna1 = "AGCT"
        dna2 = "AGCT"

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        self.assertEqual(result.aligned_dna1, "AGCT")
        self.assertEqual(result.aligned_dna2, "AGCT")
        self.assertEqual(result.final_score, 4.0)  # 4 matches * match_score

    def test_smith_waterman_subsequence(self):
        """Test SW alignment finding local subsequence."""
        dna1 = "GGGTCTA"
        dna2 = "TCTAGG"

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        # Should find "TCTA" as the best local alignment
        self.assertEqual(result.aligned_dna1, "TCTA")
        self.assertEqual(result.aligned_dna2, "TCTA")
        self.assertEqual(result.final_score, 4.0)  # 4 matches * match_score

    def test_smith_waterman_with_mismatches(self):
        """Test SW alignment with mismatches."""
        dna1 = "ACGTACGT"
        dna2 = "ACTTACGT"

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        # Should find two local alignments "ACT" and "ACGT"
        # Algorithm typically returns the first best scoring alignment
        self.assertEqual(len(result.aligned_dna1), len(result.aligned_dna2))
        self.assertGreater(result.final_score, 0)

    def test_smith_waterman_no_similarity(self):
        """Test SW alignment with completely different sequences."""
        dna1 = "AAAA"
        dna2 = "TTTT"

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        # Should find no significant local alignment
        self.assertEqual(result.aligned_dna1, "")
        self.assertEqual(result.aligned_dna2, "")
        self.assertEqual(result.final_score, 0.0)

    def test_needleman_wunsch_long_sequences(self):
        """Test NW alignment with longer, more complex sequences."""
        dna1 = "ATGGCCTCAAGGAGTCGCTGC"
        dna2 = "ATGCCTCGAGGAGTAGCTGC"

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(len(result.aligned_dna1), len(result.aligned_dna2))
        self.assertTrue('-' in result.aligned_dna1 or '-' in result.aligned_dna2)
        self.assertTrue(all(c in 'ACGT-' for c in result.aligned_dna1))
        self.assertTrue(all(c in 'ACGT-' for c in result.aligned_dna2))

    def test_needleman_wunsch_repeated_sections(self):
        """Test NW alignment with repeated sequence sections."""
        dna1 = "ACACACACGT"
        dna2 = "ACACACGT"

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        # Verify that repeating patterns are aligned correctly
        self.assertTrue(result.aligned_dna1.startswith("ACACAC"))
        self.assertTrue(result.aligned_dna2.startswith("ACACAC"))

    def test_needleman_wunsch_all_gaps_one_side(self):
        """Test NW alignment where one sequence must be all gaps."""
        dna1 = "ACGT"
        dna2 = ""

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(result.aligned_dna2, "-" * len(dna1))
        self.assertEqual(result.final_score, len(dna1) * self.gap_penalty)

    def test_needleman_wunsch_different_scoring(self):
        """Test NW alignment with different scoring parameters."""
        dna1 = "ACGT"
        dna2 = "ACTT"

        custom_score_matrix = create_alignment_score_matrix(
            match_score=2,
            mismatch_score=-2
        )

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=custom_score_matrix,
            gap_penalty=-2
        )

        result = next(needleman_wunsch(nw_input))

        # Score should be: 3 matches (6) + 1 mismatch (-2) = 4
        self.assertEqual(result.final_score, 4.0)

    def test_smith_waterman_overlapping_regions(self):
        """Test SW alignment with multiple possible overlapping regions."""
        dna1 = "AAACTGCTGCAA"
        dna2 = "TTCTGCTGCTT"

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        # Should find "CTGCTGC" as the best local alignment
        self.assertTrue("CTGCTGC" in result.aligned_dna1)
        self.assertTrue("CTGCTGC" in result.aligned_dna2)

    def test_smith_waterman_asymmetric_sequences(self):
        """Test SW alignment with very different length sequences."""
        dna1 = "AAAAACTGCTAAAAA"  # Target in middle
        dna2 = "CTGCT"            # Short query

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        self.assertEqual(result.aligned_dna1, "CTGCT")
        self.assertEqual(result.aligned_dna2, "CTGCT")
        self.assertEqual(result.final_score, 5.0)  # 5 matches

    def test_smith_waterman_multiple_matches(self):
        """Test SW alignment with multiple equally good matches."""
        dna1 = "ATATATATATAT"
        dna2 = "ATAT"

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = smith_waterman(sw_input)

        # Should find one of the "ATAT" matches
        self.assertEqual(len(result.aligned_dna1), 4)
        self.assertEqual(result.aligned_dna1, "ATAT")
        self.assertEqual(result.aligned_dna2, "ATAT")
        self.assertEqual(result.final_score, 4.0)

    def test_score_matrix_symmetry(self):
        """Test that score matrices are computed symmetrically."""
        dna1 = "ACGT"
        dna2 = "TGCA"

        # Test both directions
        nw_input_1 = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        nw_input_2 = NeedlemanWunschInput(
            dna1=dna2,
            dna2=dna1,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result1 = next(needleman_wunsch(nw_input_1))
        result2 = next(needleman_wunsch(nw_input_2))

        self.assertEqual(result1.final_score, result2.final_score)

    def test_real_world_sequences(self):
        """Test alignment with real-world-like DNA sequences."""
        # Fragment of a human gene sequence
        dna1 = "ATGGAGTCTCCGCAGGGTCAGGAGTCCCTGAGC"
        # Same sequence with mutations and gaps
        dna2 = "ATGGAGTATCCGCAGGTCAGGTCCCTGAGC"

        # Test both algorithms
        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        sw_input = SmithWatermanInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        nw_result = next(needleman_wunsch(nw_input))
        sw_result = smith_waterman(sw_input)

        # Verify basic properties
        self.assertEqual(len(nw_result.aligned_dna1), len(nw_result.aligned_dna2))
        self.assertGreater(sw_result.final_score, 0)

        # SW should find at least as good a local alignment as NW
        self.assertGreaterEqual(sw_result.final_score, nw_result.final_score)

    def test_boundary_conditions(self):
        """Test various boundary conditions."""
        # Single character sequences
        dna1 = "A"
        dna2 = "A"

        nw_input = NeedlemanWunschInput(
            dna1=dna1,
            dna2=dna2,
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(result.aligned_dna1, "A")
        self.assertEqual(result.aligned_dna2, "A")
        self.assertEqual(result.final_score, 1.0)

        # Empty sequences
        nw_input = NeedlemanWunschInput(
            dna1="",
            dna2="",
            alignment_score_matrix=self.alignment_score_matrix,
            gap_penalty=self.gap_penalty
        )

        result = next(needleman_wunsch(nw_input))

        self.assertEqual(result.aligned_dna1, "")
        self.assertEqual(result.aligned_dna2, "")
        self.assertEqual(result.final_score, 0.0)


if __name__ == '__main__':
    unittest.main()
