# 📰 Fake News Detection using BERT

This project leverages **BERT** (Bidirectional Encoder Representations from Transformers) to classify news articles as **real** or **fake**. By fine-tuning a pre-trained BERT model on a labeled dataset, the system learns to identify the subtle linguistic patterns that differentiate trustworthy journalism from misleading content.

---

## 🚀 Features and Approach

* **BERT Transformer Model**
  The core of this project is the `bert-base-uncased` model from Hugging Face Transformers. Fine-tuning this pre-trained model allows it to adapt quickly to the fake news classification task without training from scratch.

* **Transfer Learning**
  A linear classification head is added on top of the \[CLS] token of BERT for binary classification. The entire model is fine-tuned to optimize performance on the fake news detection task.

* **High Accuracy**
  The model achieves over **95% accuracy** on benchmark datasets like ISOT, significantly outperforming traditional machine learning approaches.

* **Context-Aware Understanding**
  Unlike bag-of-words or RNN models, BERT captures full bidirectional context, making it especially effective in understanding the semantics of news text.

---

## 📊 Dataset

* **Source**: [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)
  Compiled by the ISOT Lab at the University of Victoria.

* **Content**:

  * `True.csv` – Real news articles
  * `Fake.csv` – Fake news articles
  * (Alternatively, a combined `News.csv` with a `label` column may be used)

* **Size**: Approximately 45,000 articles

  * \~21,000 real
  * \~23,000 fake

* **Format**: CSV files containing full article text (and optionally titles or subjects). Only textual content is used; no images or metadata.

---

## 🛠️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/UNIVPM-DataScience/Bert-fake-news.git
   cd Bert-fake-news
   ```

2. **Create a Python Environment (Optional but Recommended)**

   ```bash
   conda create -n fake-news-bert python=3.9
   conda activate fake-news-bert
   ```

3. **Install Dependencies**

   ```bash
   pip install torch torchvision torchaudio
   pip install transformers
   pip install pandas scikit-learn numpy
   pip install jupyterlab notebook
   ```

4. **Ensure Dataset Availability**
   The `Dataset/` directory should contain `True.csv` and `Fake.csv`, or a combined dataset. If not, download the ISOT dataset and place the files in the folder manually.

5. **Launch the Jupyter Notebook**

   ```bash
   jupyter notebook FakeNews_Bert.ipynb
   ```

---

## 📌 Usage Instructions

### Run the Notebook

Open `FakeNews_Bert.ipynb` and execute each cell in order:

1. **Data Loading & Preprocessing**
   Load the dataset, clean the text, encode the labels, and split the data into training and test sets.

2. **Model Initialization**
   Load `bert-base-uncased` with a binary classification head and its tokenizer.

3. **Fine-Tuning the Model**
   Train the model using your dataset (GPU strongly recommended). You can configure:

   * Number of epochs (e.g., 2–4)
   * Batch size (e.g., 16 or 32)
   * Learning rate (e.g., 2e-5)

4. **Model Evaluation**
   Evaluate performance on the test set using accuracy, confusion matrix, classification report, etc.

5. **Custom Text Prediction**
   Provide a new article or headline to get predictions ("Fake" or "Real").

### Save and Reload the Model

To save:

```python
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
```

To reload:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')
```

---

## 📁 Project Structure

```
Bert-fake-news/
│
├── FakeNews_Bert.ipynb         # Main Jupyter notebook
├── Dataset/                    # Contains the dataset
│   ├── True.csv
│   └── Fake.csv
└── saved_model/ (optional)     # Saved model after fine-tuning
```

---

## 💡 Possible Extensions

* Use **DistilBERT** or **RoBERTa** for better speed or accuracy
* Build a **Flask** or **Streamlit** web app for deployment
* Add **attention visualization** to explain predictions
* Perform **hyperparameter tuning** to improve performance

---

## 🧪 Troubleshooting

* **Out of Memory (OOM)**: Try reducing the batch size or using `distilbert-base-uncased`
* **Version Issues**: Check compatibility on [Hugging Face Docs](https://huggingface.co/docs/transformers/index)
* **Slow CPU Training**: Run on Google Colab or a machine with GPU acceleration

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
