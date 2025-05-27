# Log Analyzer

An intelligent log analysis tool that uses machine learning to detect anomalies in log files. Features template mining with Drain3, statistical analysis with Isolation Forest, and advanced ML analysis using Gemma-2B or cloud APIs.

## Features

- **🧠 Adaptive Context-Aware Analysis**: Discovers log structure first, then adapts analysis accordingly (RECOMMENDED)
- **📋 Template Mining**: Uses Drain3 algorithm to identify log patterns
- **📊 Statistical Analysis**: Isolation Forest for detecting statistical anomalies  
- **🤖 ML Analysis**: Local Gemma-2B model with cloud API fallback
- **💰 Budget Management**: Tracks API costs and prevents overruns
- **🎨 Rich Output**: Colorized terminal output and structured JSON reports
- **⚡ Async Processing**: Fast cloud-only mode with parallel requests

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Authentication (for Gemma-2B model)

```bash
# Login to Hugging Face (required for Gemma-2B)
huggingface-cli login

# Or set token as environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### Usage

#### 🧠 Adaptive Mode (Recommended)
The smartest approach that learns your log structure first:

```bash
# Adaptive context-aware analysis
python main.py --input sample_logs.txt --adaptive --output results.json

# With debug information
python main.py --input sample_logs.txt --adaptive --debug
```

**How Adaptive Mode Works:**
1. **Phase 1**: Samples 10 random lines from different parts of your log file
2. **Phase 2**: Uses Gemma-2B to understand log type, format, and structure  
3. **Phase 3**: Creates tailored prompts based on discovered context
4. **Phase 4**: Analyzes each line with context-aware prompts for better accuracy

#### 🏠 Normal Mode (Legacy)
Traditional approach with local analysis + cloud fallback:

```bash
# Standard analysis
python main.py --input sample_logs.txt --output results.json

# With custom threshold
python main.py --input sample_logs.txt --threshold 0.7
```

#### 🌩️ Cloud-Only Mode (Legacy)
Fast parallel processing using only cloud APIs:

```bash
# Cloud-only with async processing
python main.py --input sample_logs.txt --cloud-only --output results.json
```

## Architecture

### Core Components

1. **`parser.py`** - Drain3-based log template mining
2. **`stats.py`** - Statistical anomaly detection with Isolation Forest
3. **`ml_analyzer.py`** - Advanced ML analysis with adaptive context discovery
4. **`reporter.py`** - Colorized output and JSON reporting
5. **`budget_guard.py`** - API cost management and monitoring

### Analysis Pipeline

```
Log File → Template Mining → Statistical Analysis → ML Analysis → Report
           (Drain3)        (Isolation Forest)   (Gemma-2B/Cloud)
```

**Adaptive Pipeline:**
```
Log File → Sample Lines → Context Discovery → Adaptive Prompts → Line Analysis → Report
          (10 random)    (Gemma-2B)         (Tailored)       (Context-aware)
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# OpenAI API (for cloud fallback)
OPENAI_API_KEY=your_openai_api_key

# Hugging Face (for Gemma-2B)
HUGGINGFACE_HUB_TOKEN=your_hf_token

# Budget settings
MAX_BUDGET=45.0
BUDGET_BUFFER=5.0
```

### Advanced Configuration

The tool automatically configures itself, but you can customize:

- **Anomaly threshold**: `--threshold 0.6` (0.0-1.0)
- **Model selection**: Currently uses `google/gemma-2b`
- **Budget limits**: Set in environment or budget_guard.py

## Example Output

### Adaptive Mode Output

```
🧠 ADAPTIVE MODE: Context-aware analysis with Gemma-2B
💡 This discovers log structure first, then adapts analysis accordingly

Phase 1: Discovering log context...
Context discovered: Web Server logs
Phase 2: Adaptive line-by-line analysis...

============================================================
📊 LOG ANALYSIS SUMMARY
============================================================

🔍 DISCOVERED LOG CONTEXT:
   Log Type: Web Server
   Format: Apache Common Log Format
   Severity Levels: ERROR, WARN, INFO
   Key Fields: timestamp, method, url, status, size
   Context Discovery: ✅ Success

🔧 Analysis Method: Adaptive Context Aware

📈 STATISTICS:
   Total lines analyzed: 1,000
   Normal entries: 987
   Anomalies detected: 13
   Anomaly rate: 1.30%

🚨 HIGH SEVERITY ANOMALIES (3):
   Line 245: HTTP 500 Internal Server Error
   Line 456: Connection timeout after 30s
   Line 789: Authentication failed for user admin

✅ Context discovery successful!
🚨 Found 13 anomalies
```

## Model Information

### Current Models

- **Local**: `google/gemma-2b` (2B parameters, efficient, log-aware prompting)
- **Cloud Fallback**: OpenAI GPT-3.5-turbo (when local fails or budget allows)

### Hardware Requirements

- **CPU Only**: 8GB+ RAM (slower but works)
- **GPU Recommended**: 6GB+ VRAM for faster processing
- **Storage**: ~5GB for model download

## Performance

### Adaptive Mode Benefits

- **Better Accuracy**: Context-aware prompts reduce false positives
- **Adaptive Learning**: Tailors analysis to specific log types
- **Structure Recognition**: Understands log format automatically
- **Intelligent Prompting**: Uses discovered patterns for better detection

### Speed Comparison

- **Adaptive Mode**: ~100-200 lines/minute (includes context discovery)
- **Normal Mode**: ~50-100 lines/minute (template mining + ML)  
- **Cloud-Only**: ~200-500 lines/minute (parallel async requests)

## Troubleshooting

### Common Issues

1. **Gemma-2B Access Denied**
   ```bash
   huggingface-cli login
   # Make sure you accepted the license at https://huggingface.co/google/gemma-2b
   ```

2. **Memory Issues**
   - Reduce batch size in cloud-only mode
   - Use CPU instead of GPU: set `CUDA_VISIBLE_DEVICES=""`

3. **Budget Exceeded**
   - Check `budget_usage.json`
   - Adjust `MAX_BUDGET` in `.env`

4. **Context Discovery Fails**
   - Tool automatically falls back to heuristic analysis
   - Check debug logs for details

### Debug Mode

```bash
python main.py --input sample_logs.txt --adaptive --debug
```

## Future Improvements

- **LogBERT Integration**: Specialized log model when available
- **Fine-tuning**: Custom model training on user data
- **Real-time Analysis**: Stream processing capabilities
- **Advanced Metrics**: Precision/recall evaluation
- **Custom Rules**: User-defined anomaly patterns

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request