# 📰 Fake News Classification with BERT

This project was developed as part of the **Data Science course** at the **Università Politecnica delle Marche**, Academic Year 2024–2025.

## 📌 Project Description

This repository tackles the problem of **automatic fake news classification** using advanced **Natural Language Processing (NLP)** techniques based on **Transformer architectures**, specifically **BERT (Bidirectional Encoder Representations from Transformers)**.

By fine-tuning a pre-trained BERT model on the ISOT Fake News dataset, we achieve high-accuracy classification between real and fake news articles.

---

## 🎯 Objectives

- Build an NLP system capable of identifying fake news with high precision.
- Apply fine-tuning on a pre-trained BERT model using labeled news data.
- Compare BERT’s performance with other Transformer models.

---

## 📚 Dataset

- **Source:** ISOT Fake News Dataset (University of Victoria)
- **Size (post-cleaning):** ~18,000 articles
- **Class distribution:** 56.7% Fake, 43.3% Real
- **Fields:** `id`, `title`, `author`, `text`, `label` (0 = real, 1 = fake)

---

## 🧹 Data Preprocessing

- Text cleaning (special characters, repeated characters, lowercasing)
- Tokenization using `bert-base-uncased`
- Truncation to `max_length = 512` tokens
- Dataset split: **70% train**, **10% validation**, **20% test**

---

## 🧠 Model Architecture

- **Base model:** `bert-base-uncased` (Hugging Face)
- **Architecture:** BERT encoder + binary classification head
- **Training Parameters:**
  - Batch size: 16–32
  - Learning rate: `2e-5`
  - Epochs: 3 (with early stopping)
  - Optimizer: AdamW

---

## 🧪 Development Environment

- Language: **Python 3.9**
- Frameworks/Libraries: `PyTorch`, `Transformers` (Hugging Face), `scikit-learn`, `pandas`, `numpy`
- IDE: Jupyter Notebook
- Hardware: **GPU recommended** for training

---

## 📊 Results

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 95.2%     |
| Precision  | 94.8%     |
| Recall     | 95.5%     |
| F1-Score   | 95.1%     |
| AUC (ROC)  | 0.98      |

📌 **Transformer Benchmark Results:**

| Model             | Accuracy | F1-Score |
|------------------|----------|----------|
| DistilBERT       | 92.5%    | 93.3%    |
| RoBERTa          | 90.3%    | 90.1%    |
| ALBERT           | 89.6%    | 89.5%    |
| MobileBERT       | 87.1%    | 86.1%    |

DistilBERT showed the best trade-off between accuracy and computational efficiency.

--- 

## ⚠️ Limitations
Domain specificity: The model was trained on Reuters-style articles and may not generalize well to social media or informal content.

Data bias: The dataset may reflect inherent biases due to source selection.

Token limit: BERT’s 512-token input limit could truncate long articles, possibly omitting critical information.

Hardware constraints: Model training was limited by available computational resources.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repository
2. Create a new branch (`feature/my-feature`)
3. Commit your changes
4. Push the branch and open a pull request

> For large feature changes, please open an issue first to discuss ideas and improvements.

---

## 📄 License

**License**: TBD
This project does not yet have a formal license file. Assume academic and personal usage only unless otherwise noted. Contact the maintainers for commercial or extended use.

---

## 📚 References

* [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
* [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Colab Demo for BERT Fine-Tuning](https://colab.research.google.com/)

---


> 🧠 Developed as part of an educational data science project by the UNIVPM Data Science Group.
