# 🎯 OdysseyAI Question-Answer Evaluator

A Streamlit-based web application for evaluating question-answer pairs using OdysseyAI agents and Groq's LLM evaluation capabilities.

## 📋 Overview

This tool allows you to:
- Upload Excel files containing question-answer pairs
- Evaluate answers using OdysseyAI agents or default chat
- Get AI-powered accuracy scoring using Groq's LLM
- Export detailed evaluation results
- Support for both production and staging environments

**Available in two formats:**
- 🖱️ **Simple Executable**: `Odyssey_QA_Evaluator` - No technical setup required
- 🐍 **Python Application**: Full source code for developers and customization

## ✨ Features

- **Multi-Environment Support**: Switch between production and staging OdysseyAI environments
- **Agent Integration**: Support for various OdysseyAI agents including parameter-based and message-based agents
- **Automated Evaluation**: Uses Groq's LLM to score answer accuracy (0-100 scale)
- **Batch Processing**: Process multiple Q&A pairs from Excel files
- **Detailed Analytics**: Comprehensive statistics and evaluation metrics
- **Export Results**: Download evaluated results as Excel files
- **Real-time Progress**: Live progress tracking during evaluation

## 🚀 Quick Start

### For Non-Technical Users (Recommended)

**Simple Setup:**
1. Download this repository as a ZIP file (click the green "Code" button → "Download ZIP")
2. Extract/unzip the downloaded file to a folder on your computer
3. Navigate to the extracted folder and double-click `Odyssey_QA_Evaluator` to run
4. Your web browser will automatically open with the evaluation tool
5. No Python installation or technical setup required!

### For Developers/Technical Users

### Prerequisites

- Python 3.7+
- Groq API key
- OdysseyAI API key
- OdysseyAI Workspace ID and User ID

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd odyssey-evaluator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app_ui.py
```

## 📦 Dependencies

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
pandas>=1.5.0
requests>=2.28.0
groq>=0.4.0
openpyxl>=3.1.0
```

## 🔧 Configuration

### API Keys
- **Groq API Key**: Get from [Groq Console](https://console.groq.com/)
- **OdysseyAI API Key**: Obtain from your OdysseyAI account

### Environment Settings
- **Production**: `https://app.odysseyai.ai/api`
- **Staging**: `https://app.stage.odysseyai.ai/api`

### Required Information
- **Workspace ID**: Your OdysseyAI workspace identifier
- **User ID**: Your OdysseyAI user identifier

## 📊 Input File Format

Your Excel file should contain at minimum:
- **Question Column**: Contains the questions to be answered
- **Answer Column**: Contains the expected/reference answers

Example:
| Question | Expected Answer |
|----------|----------------|
| What is the capital of France? | Paris |
| How many continents are there? | Seven |

## 🤖 Agent Types

### Message-Based Agents
- Use questions directly as input
- Examples: `agenframe`, `dqp-agent`
- No parameter configuration required

### Parameter-Based Agents
- Require input parameter configuration
- Map Excel columns to agent parameters
- Support both fixed values and column mapping

## 📈 Evaluation Metrics

The tool provides comprehensive evaluation metrics:

- **Accuracy Score**: 0-100 scale rating
- **Correctness**: Binary yes/no classification
- **Success Rate**: Percentage of successful API calls
- **Detailed Explanations**: AI-generated evaluation reasoning
- **Difference Analysis**: Key differences between expected and actual answers

## 🔄 Workflow

1. **Environment Selection**: Choose production or staging
2. **API Configuration**: Enter required API keys and IDs
3. **File Upload**: Upload Excel file with Q&A pairs
4. **Column Mapping**: Select question and answer columns
5. **Agent Selection**: Choose appropriate OdysseyAI agent
6. **Parameter Configuration**: Set up agent parameters (if required)
7. **Evaluation**: Process all Q&A pairs
8. **Results Analysis**: Review metrics and statistics
9. **Export**: Download evaluated results

## 📁 Output Format

The tool generates an Excel file with additional columns:
- `OdysseyAI_Answer`: Response from OdysseyAI
- `Is_Correct`: yes/no correctness classification
- `Accuracy_Score`: Numerical score (0-100)
- `Evaluation_Explanation`: Detailed evaluation reasoning
- `Differences`: Key differences identified
- `API_Status`: Success/Failed status

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python with requests for API calls
- **Evaluation**: Groq's Llama model for scoring
- **Data Processing**: Pandas for Excel handling

### Rate Limiting
- 2-second delay between API calls
- Configurable timeout (320 seconds for chat API)
- Graceful error handling

### Error Handling
- API failure detection and logging
- Partial result preservation
- Detailed error messages

## 🔒 Security

- API keys handled securely (password input fields)
- No persistent storage of sensitive data
- Environment-based URL configuration

## 📝 Usage Examples

### Basic Evaluation
1. Upload Excel with "Question" and "Answer" columns
2. Select default chat (no agent)
3. Run evaluation

### Agent-Based Evaluation
1. Upload Excel file
2. Select specific OdysseyAI agent
3. Configure required parameters
4. Run evaluation

### Parameter Mapping
- **Fixed Value**: Use same value for all rows
- **Column Mapping**: Map Excel column to agent parameter

## 🐛 Troubleshooting

### For Executable Version Users
- **Application won't start**: Try running as administrator (right-click → "Run as administrator")
- **Browser doesn't open**: Manually open your browser and go to `http://localhost:8501`
- **Antivirus blocking**: Add the application to your antivirus whitelist
- **Windows security warning**: Click "More info" → "Run anyway" (the application is safe)

### For Python Version Users

### Common Issues
- **API Errors**: Check API keys and network connectivity
- **Agent Parameters**: Ensure all required parameters are configured
- **File Format**: Verify Excel file structure and column names
- **Environment**: Confirm correct environment selection

### Debug Information
The application provides detailed debug information including:
- API response details
- Parameter configuration
- Processing status
- Error messages