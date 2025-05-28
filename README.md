# Log-Based Intrusion Detection with LLMs

This project implements an intrusion detection system that analyzes log entries using Large Language Models (LLMs). The system supports multiple LLM providers (Ollama, OpenAI, Gemini) to analyze and classify log entries for potential security threats.

## Features

- Multi-provider LLM support:
  - Ollama (default, local deployment)
  - OpenAI (API key required)
  - Google's Gemini (API key required)
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
- One or more of the following LLM providers:
  - Ollama installed locally (for Ollama provider)
  - OpenAI API key (for OpenAI provider)
  - Google Cloud API key (for Gemini provider)
- uv package manager (recommended) or pip

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

2. Install dependencies using uv (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your chosen LLM provider:

   For Ollama (local deployment):
   ```bash
   # Install Ollama if not already installed (macOS/Linux)
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull the Gemma model
   ollama pull gemma:12b
   ```

   For OpenAI/Gemini:
   - Obtain the necessary API keys from the respective provider
   - Use the `--api-key` flag when running the detection script or agent

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

You can run the intrusion detection system in two ways:

1. **Real-time Monitoring with Agent**

The agent continuously monitors a directory for new or modified log files and analyzes them in real-time:

```bash
# Using Ollama (default)
python src/agent.py \
    --watch data/logs/ \
    --output results/

# Using OpenAI
python src/agent.py \
    --watch data/logs/ \
    --output results/ \
    --provider openai \
    --api-key "your-openai-key" \
    --model-name "text-embedding-3-large"

# Using Gemini
python src/agent.py \
    --watch data/logs/ \
    --output results/ \
    --provider gemini \
    --api-key "your-google-key" \
    --model-name "embedding-001"
```

The agent will:
- Watch for new .csv log files in the specified directory
- Analyze existing log files when started
- Process new or modified files automatically
- Generate analysis summaries and alerts
- Save results to specified output directory

2. **Batch Analysis of Log Files**

For analyzing specific log files:

```bash
# Using Ollama (default)
python src/run_detection.py \
    --input data/logs/sample.csv \
    --output results/analysis.csv

# Using OpenAI
python src/run_detection.py \
    --input data/logs/sample.csv \
    --output results/analysis.csv \
    --provider openai \
    --api-key "your-openai-key"

# Using Gemini
python src/run_detection.py \
    --input data/logs/sample.csv \
    --output results/analysis.csv \
    --provider gemini \
    --api-key "your-google-key"
```

### Command Line Parameters

Common parameters for both modes:
- `--batch-size`: Number of logs to process at once (default: 8)
- `--threshold`: Confidence threshold for high-risk classification (default: 0.7)
- `--output`: Directory to save analysis results (optional)
- `--provider`: LLM provider to use ("ollama", "openai", or "gemini")
- `--model-name`: Model name for the chosen provider
- `--api-key`: API key for OpenAI or Gemini (not needed for Ollama)

Agent-specific parameters:
- `--watch`: Directory to monitor for log files (required)

Detection-specific parameters:
- `--input`: Path to the CSV file containing log entries (required)

### Output Format

The system generates several types of output:
1. CSV files with analysis results including:
   - Predictions (0: benign, 1: malicious)
   - Confidence scores
   - Threat levels (Benign/Suspicious/High Risk)

2. JSON summary reports containing:
   - Total logs analyzed
   - High risk entry counts and percentages
   - Suspicious entry counts and percentages
   - Timestamp and file information

3. Real-time alerts for high-risk entries (in agent mode)

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
   - Provider-specific embeddings:
     - Ollama: 4096 dimensions
     - OpenAI: 3072 dimensions
     - Gemini: 768 dimensions
   - Linear classifier for binary classification
   - PyTorch-based training and inference

## License

Apache License 2.0