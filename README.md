# 🕵️‍♂️ Fake Job Posting Detection: An NLP Architecture Comparison

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface&logoColor=white)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)

## 📌 Project Overview
Identifying fraudulent job postings is a complex natural language processing (NLP) challenge, largely due to extreme class imbalance and the nuanced linguistic patterns of scammers. 

This project implements a comprehensive deep learning pipeline to detect fake job postings using the **EMSCAD dataset**. Rather than relying on a single model, this project evaluates and compares **10 distinct architectures** to analyze the trade-offs between sequential processing, parallel feature extraction, and modern self-attention mechanisms.

## 📊 Dataset
* **Source:** EMSCAD Fake Job Postings Dataset
* **Size:** ~17,880 records
* **Target Variable:** `fraudulent` (Binary: 0 = Real, 1 = Fake)
* **Challenge:** Severe class imbalance (~95% Real, ~5% Fake) mitigated via Cross-Entropy class weighting.
* **Feature Engineering:** Text-heavy features (`title`, `company_profile`, `description`, `requirements`, `benefits`) were concatenated into a single master sequence.

## 🧠 Models Implemented
The pipeline evaluates the evolution of NLP architectures across three categories:

1. **Recurrent Models:** Simple RNN, LSTM, GRU, BiLSTM, LSTM with Attention
2. **Convolutional & Character Models:** 1D CNN (TextCNN), Character-Level Deep Model (Char-CNN-LSTM)
3. **Transformer Models:** BERT (bert-base-uncased), T5 (t5-small), DistilGPT-2

## 🏆 Key Results & Performance

*Note: Models were evaluated based on F1-Score to account for the heavy class imbalance.*

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT** | 0.9850 | 1.0000 | 0.7621 | **0.8646** | 993.4s |
| **1D CNN (TextCNN)** | 0.9780 | 0.8841 | 0.8362 | **0.8595** | 16.5s |
| **DistilGPT-2** | 0.9810 | 0.9412 | 0.7634 | **0.8430** | 412.3s |
| **T5** | 0.9805 | 0.9388 | 0.7634 | **0.8421** | 501.2s |
| **BiLSTM** | 0.9520 | 0.6512 | 0.8710 | **0.7450** | 82.1s |

### 📈 Visualizing the Trade-offs

*(Upload your generated images to the repo and link them here)*
![Model Comparison Bar Chart](model_comparison_bars.png)
![Metrics Heatmap](model_heatmap.png)

## 💡 Key Engineering Insights
1. **The Vanishing Gradient:** Traditional Simple RNNs failed dramatically (F1 = 0.1521) due to the vanishing gradient problem on lengthy job descriptions. Introducing Gating (LSTM/GRU) and Bidirectional context (BiLSTM) significantly bridged this gap.
2. **Transformer Dominance:** BERT achieved a perfect Precision score (1.0) and the highest F1-Score (0.86) by utilizing bidirectional self-attention to understand the global context of the text simultaneously.
3. **Text-to-Text Versatility:** T5 successfully handled the task via text generation (outputting the strings "real" or "fake") rather than standard mathematical logits, proving the flexibility of a unified text-to-text framework.
4. **The Efficiency Trade-off:** While Transformers achieved the highest accuracy, the **1D TextCNN** achieved a highly competitive F1-Score (0.8595) while training in just **16.5 seconds** (60x faster than BERT). This proves parallelized feature extraction remains highly viable for resource-constrained deployments.

## ⚙️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Fake-Job-Detection-NLP.git](https://github.com/YourUsername/Fake-Job-Detection-NLP.git)
   cd Fake-Job-Detection-NLP
