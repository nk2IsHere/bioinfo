# DNA Sequence Alignment Algorithms Implementation

This project implements two DNA sequence alignment algorithms:
- Needleman-Wunsch algorithm for global sequence alignment
- Smith-Waterman algorithm for local sequence alignment

## Project Structure

```
.
├── data/
│   ├── nw_alignment_score_matrix.csv
│   ├── nw_output.txt
│   ├── sw_alignment_score_matrix.csv
│   └── sw_output.txt
├── main.ipynb
├── needleman_wunsch.py
├── smith_waterman.py
└── utils.py
```

## Requirements

The project requires the following Python packages:
- pandas
- matplotlib

## Components

### Main Components

1. `needleman_wunsch.py`: Implementation of the Needleman-Wunsch algorithm for global alignment
    - `NeedlemanWunschInput`: Input configuration class
    - `NeedlemanWunschOutput`: Output results class
    - `needleman_wunsch()`: Main algorithm implementation
    - `draw_needleman_wunsch()`: Visualization function

2. `smith_waterman.py`: Implementation of the Smith-Waterman algorithm for local alignment
    - `SmithWatermanInput`: Input configuration class
    - `SmithWatermanOutput`: Output results class
    - `smith_waterman()`: Main algorithm implementation

3. `utils.py`: Utility functions and classes
    - `AlignmentScoreMatrix`: Score matrix implementation
    - `TracebackAction`: Enum for alignment actions
    - Helper functions for creating and managing alignment matrices

### Configuration

The project uses two types of configuration files:
1. Alignment score matrices (CSV files):
    - `nw_alignment_score_matrix.csv`: For Needleman-Wunsch algorithm
    - `sw_alignment_score_matrix.csv`: For Smith-Waterman algorithm

2. Output files:
    - `nw_output.txt`: Global alignment results
    - `sw_output.txt`: Local alignment results

## Usage

### Basic Usage

1. Open `main.ipynb` in a Jupyter notebook environment.

2. Configure the input parameters:
```python
# Needleman-Wunsch configuration
nw_maximal_final_alignments_count = 10
nw_gap_penalty = -2
nw_dna1 = "TGCTCGTA"
nw_dna2 = "TTCATA"

# Smith-Waterman configuration
sw_gap_penalty = -2
sw_dna1 = "TGCTCGTA"
sw_dna2 = "TTCATA"
```

3. Run the cells to execute the alignments.

### Example Output

Needleman-Wunsch (Global Alignment):
```
TGCTCG-TA    T--TC-ATA    17.0
TGCTC-GTA    T--TCA-TA    17.0
TGCTCGTA     T--TCATA     17.0
```

Smith-Waterman (Local Alignment):
```
TGCTCGTA    T--TCATA    17
```
