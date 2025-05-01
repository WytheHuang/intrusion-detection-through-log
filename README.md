# Log-Based Intrusion Detection with LLMs

This project implements an intrusion detection system that analyzes log entries using Large Language Models (LLMs) through the Ollama API. The system uses Gemma 12B to analyze and classify log entries for potential security threats.

## Features

- Advanced log entry analysis using Ollama's Gemma 12B model
- Multi-level threat classification (High Risk/Suspicious/Benign)
- Sophisticated log preprocessing including:
  - Timestamp extraction
  - IP address detection
  - Security-relevant keyword identification
- Confidence scoring for threat detection
- Batch processing capabilities
- Command-line interface for easy analysis

## Prerequisites

- Python 3.12+
- Ollama installed and running locally
- Required Python packages (install via `pip install -r requirements.txt`):
  - torch
  - numpy
  - pandas
  - scikit-learn
  - tqdm
  - ollama
  - argparse

## Project Structure

```
.
├── data/               # Directory for log data
├── src/
│   ├── data_processor.py  # Data loading and processing
│   ├── model.py          # Model architecture and classifier
│   ├── predict.py        # Prediction utilities
│   └── run_detection.py  # Main detection script
├── docs/               # Documentation
└── tests/              # Test files
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and running on your system with the Gemma 12B model:
   ```bash
   ollama pull gemma:12b
   ```

## Usage

### Running Intrusion Detection

To analyze log files for potential security threats:

```bash
python src/run_detection.py \
    --input path/to/logs.csv \
    --output results.csv \
    --batch-size 8 \
    --threshold 0.7
```

Parameters:
- `--input`: Path to the CSV file containing log entries (required)
- `--output`: Path to save the analysis results (optional)
- `--batch-size`: Number of logs to process at once (default: 8)
- `--threshold`: Confidence threshold for high-risk classification (default: 0.7)

### Input Data Format

The input CSV file should contain:
- A column named 'log_text' containing the log messages, or
- The first column will be used by default if 'log_text' is not present

### Output Format

The analysis results will include:
- Original log entry
- Threat classification (High Risk/Suspicious/Benign)
- Confidence score
- Detection timestamp

## Model Architecture

The system uses a two-stage approach:
1. Log entries are preprocessed to extract security-relevant information
2. Gemma 12B model analyzes the processed logs for threat detection

The analysis pipeline includes:
- Advanced preprocessing of log entries
- Embedding generation using Gemma 12B
- Multi-level threat classification
- Confidence scoring

## License

Apache License 2.0