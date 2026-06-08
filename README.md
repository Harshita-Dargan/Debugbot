# 🐞 DebugBot: Explainable AI-Based Python Code Debugger

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![CodeBERT](https://img.shields.io/badge/Model-CodeBERT-green)
![XAI](https://img.shields.io/badge/Explainable-AI-purple)

**An Explainable AI-powered debugging assistant that detects Python code defects and visualizes model reasoning through heatmaps and color-coded token attribution.**

</div>

---

## 📖 Overview

DebugBot is an Explainable AI (XAI) based Python debugging assistant that combines the power of transformer-based code understanding with interpretable machine learning techniques.

The system uses a fine-tuned **CodeBERT** model trained on the **CodeXGLUE** dataset to identify defects in Python code. Unlike traditional debugging models that operate as black boxes, DebugBot provides visual explanations through **heatmaps**, **color-coded token highlighting**, and **token attribution analysis**, allowing users to understand why a bug was detected.

The project aims to improve transparency, trust, and usability in AI-assisted software development.

---

## ✨ Features

- 🔍 Automated Python code defect detection
- 🤖 Fine-tuned CodeBERT transformer model
- 🧠 Explainable AI-powered predictions
- 🌡️ Heatmap visualization of code importance
- 🎨 Color-coded error localization
- 📊 Token-level attribution analysis
- ⚠️ Bug-prone region identification
- 💡 Interpretable debugging recommendations
- 📚 Educational and developer-friendly interface

---

## 🏗️ System Architecture

```text
Python Source Code
         │
         ▼
CodeBERT Tokenizer
         │
         ▼
Fine-Tuned CodeBERT
         │
 ┌───────┴────────┐
 ▼                ▼
Bug Detection   Attribution Analysis
                     │
                     ▼
          Explainability Engine
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Heatmap Generation     Color Coding
         │                       │
         └───────────┬───────────┘
                     ▼
            Debugging Insights
                     │
                     ▼
               User Output
```

---

## 🛠️ Tech Stack

### Machine Learning

- PyTorch
- Hugging Face Transformers
- CodeBERT

### Data Processing

- Python
- NumPy
- Pandas

### Explainability & Visualization

- Matplotlib
- Token Attribution Analysis
- Heatmap Visualization
- Feature Importance Mapping

### Development Tools

- Git
- GitHub

---

## 📂 Dataset

This project uses the **CodeXGLUE** benchmark dataset, a widely recognized benchmark for code intelligence and code understanding tasks.

### Dataset Components

- Python source code samples
- Defect detection datasets
- Code understanding benchmarks
- Bug classification examples

### Dataset Repository

https://github.com/microsoft/CodeXGLUE

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

- Loaded Python code samples from CodeXGLUE.
- Cleaned and normalized source code.
- Tokenized code using the CodeBERT tokenizer.
- Prepared training and validation datasets.

### 2️⃣ Model Fine-Tuning

- Loaded pretrained CodeBERT from Hugging Face.
- Fine-tuned on code defect detection tasks.
- Optimized model parameters using supervised learning.
- Evaluated model performance on validation data.

### 3️⃣ Explainability Analysis

To make predictions interpretable, DebugBot generates:

- Token-level importance scores
- Heatmap visualizations
- Color-coded code annotations
- Error localization maps

These visual explanations help users identify which code regions contributed most to the model's prediction.

### 4️⃣ Debugging Assistance

The system combines:

- Defect prediction
- Importance visualization
- Error localization
- Interpretable feedback

to provide actionable debugging insights.

---

## 🧠 Explainable AI Module

Traditional AI debugging systems often provide predictions without explaining their reasoning.

DebugBot addresses this challenge through:

### 🌡️ Heatmap Visualization

Heatmaps highlight code regions that contribute most strongly to bug predictions.

### 🎨 Color-Coded Attribution

Each token is assigned an importance score and visualized using colors.

| Color | Meaning |
|---------|----------|
| 🟥 Red | Very High Importance |
| 🟧 Orange | High Importance |
| 🟨 Yellow | Moderate Importance |
| 🟩 Green | Low Importance |

### 📊 Token Attribution Analysis

The model identifies which tokens have the greatest influence on its prediction, helping developers understand detected issues.

---

## 📸 Example

### Input Code

```python
def calculate_sum(a, b)
    return a + b
```

### DebugBot Analysis

```text
Bug Detected: Syntax Error

Importance Heatmap:

def calculate_sum(a, b)
^^^^^^^^^^^^^^^^^^^^^^
🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥

return a + b
^^^^^^
🟨🟨🟨🟨🟨🟨

Explanation:
Missing ':' after function definition.
```

### Output

✅ Bug identified

✅ Error localized

✅ Visual explanation generated

---

## 📈 Results

- Successfully fine-tuned CodeBERT for Python defect detection.
- Generated interpretable debugging explanations.
- Produced token-level attribution visualizations.
- Enabled transparent AI-assisted debugging.
- Demonstrated practical application of Explainable AI in software engineering.

---

## 🎯 Applications

### Education

- Learning programming concepts
- Understanding common coding mistakes
- Teaching AI explainability

### Software Development

- Code review assistance
- Bug localization
- AI-assisted debugging workflows

### Research

- Explainable AI (XAI)
- AI for Software Engineering
- Code Intelligence Systems

---

## 🔮 Future Enhancements

- Support for Java, C++, and JavaScript
- Real-time IDE integration
- VS Code extension
- Advanced bug localization
- LLM-powered debugging suggestions
- Interactive web dashboard
- Multi-language support
- Improved explainability techniques

---

## 🏆 Recognition

**RESCON IIT Bombay 2025**

Selected among the **Top 50 projects out of 700+ registrations**, recognizing innovation in Explainable AI and intelligent software engineering.

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/debugbot.git

cd debugbot

pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python debugbot.py
```


## 📁 Project Structure

```text
DebugBot/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│
├── notebooks/
│   ├── training.ipynb
│   ├── explainability.ipynb
│
├── models/
│   ├── fine_tuned_codebert/
│
├── visualizations/
│   ├── heatmaps/
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
│   ├── explainability.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 👩‍💻 Author

### Harshita Dargan

B.Tech Computer Science Engineering

Areas of Interest:

- Artificial Intelligence
- Explainable AI (XAI)
- Machine Learning
- Natural Language Processing
- Software Engineering

---

## 📜 License

This project is intended for educational and research purposes.

If you use this work in academic research, please cite:

- CodeBERT
- CodeXGLUE
- Hugging Face Transformers

---

## ⭐ Acknowledgements

- Microsoft Research for CodeXGLUE
- Hugging Face for Transformers
- CodeBERT Authors
- PyTorch Community

---

⭐ If you found this project useful, consider giving it a star!
