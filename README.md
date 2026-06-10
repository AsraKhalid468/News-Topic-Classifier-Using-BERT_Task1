Task 1: News Topic Classifier Using BERT
🎯 Objective
Fine-tune a pre-trained BERT transformer model (bert-base-uncased) on the AG News dataset to automatically classify news headlines and articles into 4 topic categories: World, Sports, Business, and Sci/Tech.

📂 Dataset

Name: AG News Dataset
Source: https://huggingface.co/datasets/fancyzhx/ag_news
Size: 120,000 training samples | 7,600 test samples
Classes: 4 (perfectly balanced — 30,000 samples per class)
Loading: Loaded directly via Hugging Face datasets library — no manual download needed


🛠️ Methodology / Approach
1. Data Loading & EDA

Loaded AG News dataset from Hugging Face Hub
Explored class distribution and average word count per category
Visualized distributions using bar charts

2. Preprocessing

Tokenized text using BertTokenizer with max_length=128
Applied dynamic padding via DataCollatorWithPadding
Used a subset of 6,000 training and 1,200 test samples for efficient training

3. Model Architecture

Base model: bert-base-uncased (110M parameters)
Added a 4-class linear classification head on top of [CLS] token
Pre-training heads (MLM + NSP) discarded — only encoder weights retained

4. Training

Framework: Hugging Face Trainer API
Epochs: 3
Learning rate: 2e-5
Batch size: 16 (train) / 32 (eval)
Weight decay: 0.01
Warmup steps: 100
Mixed precision (fp16) enabled on GPU

5. Evaluation

Metrics: Accuracy, Macro F1, Weighted F1
Per-class classification report
Confusion matrix heatmap
Per-class F1 bar chart

6. Deployment

Deployed using Gradio with share=True for a public URL
Returns top-4 category probabilities for any input headline


📊 Key Results
EpochTrain LossVal LossAccuracyF1 Macro10.3080.29389.8%89.7%20.2430.30590.0%89.9%30.1160.31091.4%91.4%
MetricScoreAccuracy91.4%Macro F191.4%Weighted F191.4%

🔍 Key Observations

BERT's contextual embeddings significantly outperform traditional TF-IDF + Logistic Regression baselines (~90%)
The dataset is perfectly balanced across 4 classes, making accuracy and macro F1 directly comparable
Even with only 5% of training data (6,000 samples), BERT achieves 90%+ accuracy — demonstrating strong transfer learning
Sci/Tech and Business categories show the most inter-class confusion due to overlapping vocabulary
Training loss decreased consistently across epochs while validation loss remained stable — no significant overfitting


🧰 Tools & Libraries
ToolPurposetransformersBERT model and tokenizerdatasetsAG News dataset loadingtorchDeep learning backendscikit-learnEvaluation metricsgradioModel deployment and demomatplotlibVisualizationsseabornConfusion matrix heatmappandasData manipulation

📁 Repository Structure
task1-bert-news-classifier/
│
├── task1_bert_news_classifier.ipynb   # Main notebook
├── eda_plots.png                       # EDA visualizations
├── evaluation_plots.png                # Confusion matrix + F1 chart
├── bert_ag_news_final/                 # Saved model and tokenizer
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── README.md

▶️ How to Run
bash# 1. Create environment
conda create -n bert_env python=3.10 -y
conda activate bert_env

# 2. Install dependencies
pip install transformers datasets torch scikit-learn gradio seaborn matplotlib pandas numpy

# 3. Launch Jupyter and run notebook top to bottom
jupyter notebook task1_bert_news_classifier.ipynb

🚀 Skills Demonstrated

NLP using Transformer models (BERT)
Transfer learning and fine-tuning
Evaluation metrics for multi-class text classification
Lightweight model deployment with Gradio

By Asra Khalid
DHC ID-1017
