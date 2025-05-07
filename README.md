# Log-Based Intrusion Detection with LLMs

This project implements an intrusion detection system that analyzes log entries using Large Language Models (LLMs) through the Ollama API. The system uses the Gemma model to analyze and classify log entries for potential security threats.

## Features

- Log entry analysis using Ollama's Gemma model
- Binary classification (Malicious/Benign)
- Advanced log preprocessing including:
  - Timestamp extraction and normalization
  - IP address detection
  - Security-relevant keyword identification
- Synthetic data generation capabilities
- Model fine-tuning support
- Batch processing with PyTorch
- Command-line interface for data generation and detection

## Prerequisites

- Python 3.13+
- Ollama installed and running locally
- Required Python packages (install via `pip install -r requirements.txt` or `uv sync`)

## Project Structure

```
.
├── data/               # Directory for log data
├── src/
│   ├── data_generator.py   # Synthetic log data generation
│   ├── data_processor.py   # Data loading and processing
│   ├── fine_tuning.py      # Model fine-tuning utilities
│   ├── model.py            # Model architecture and classifier
│   ├── predict.py          # Prediction utilities
│   └── run_detection.py    # Main detection script
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

3. Ensure Ollama is installed and running on your system:
   ```bash
   ollama pull gemma:12b
   ```

## Usage

### Generating Synthetic Data

To generate synthetic log data for training:

```bash
python src/data_generator.py
```

### Fine-tuning the Model

To fine-tune the model on your data:

```bash
python src/fine_tuning.py
```

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
- A 'timestamp' column with the log entry timestamp
- A 'log_text' column containing the log messages
- A 'label' column for training data (0 for benign, 1 for malicious)

## Model Architecture

The system uses a two-stage approach:
1. Log entries are preprocessed to extract security-relevant information:
   - Timestamp normalization
   - IP address extraction
   - Security keyword identification
2. The processed logs are analyzed using:
   - Gemma model for text embedding generation (4096-dimensional)
   - Linear classifier for binary classification
   - PyTorch-based training and inference

## License

Apache License 2.0