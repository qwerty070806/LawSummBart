# LawSummBart

# ⚖️ Legal Document Summarization using Extract-Then-Assign (ETA) + LoRA Fine-Tuned BART

This project implements an efficient and domain-adapted **legal document summarization system** using:

- **Extract-Then-Assign (ETA)** dataset generation  
- **BART-Large** for abstractive summarization  
- **Parameter-Efficient Fine-Tuning (LoRA)**  
- **ROUGE / BERTScore based similarity assignment**  

The goal is to produce high-quality summaries for long Indian legal documents such as court judgments, which often exceed the input limits of standard transformer models.

---

## 🚀 Features
- Builds a **7× expanded training dataset** using Extract-Then-Assign  
- Uses **seven extractive summarization techniques**  
- Assigns each ground-truth summary sentence to the most relevant extractive chunk  
- Fine-tunes BART-Large using **LoRA** (massive memory savings)  
- Achieves high performance on legal summarization tasks  
- Evaluates with ROUGE and BERTScore  
- Fully reproducible training pipeline (Kaggle-friendly)

---

## 🧠 Project Architecture



Raw Legal Document
│
▼
Preprocessing
(sentence splitting, cleaning, chunking)
│
▼
Extractive Summaries (7 methods)
TextRank • LexRank • LSA • KL-Sum • LUHN • SumBasic • SBERT
│
▼
Extract-Then-Assign (ETA)
(assign each GT summary sentence to best extractive chunk)
│
▼
Expanded Training Dataset (7×)
│
▼
LoRA Fine-Tuned BART-Large
│
▼
Final Legal Summary


---

## 📦 Dataset

We used publicly available legal datasets:

- **Rishiai Dataset** – Indian court judgments with summaries  
- **FIRE Legal Dataset** – Legal case documents and abstracts  

### Preprocessing Steps:
- Sentence splitting  
- Removing citations & noise  
- Chunking documents into <= 512 tokens  
- Preparing extractive sentences for ETA  

---

## 📘 Extractive Summarization Methods (7)

| Method | Description |
|--------|-------------|
| **TextRank** | Graph-based ranking of sentences using PageRank |
| **LexRank** | Similar to TextRank but uses cosine similarity + centrality |
| **LSA** | Topic extraction using SVD on term–sentence matrix |
| **KL-Sum** | Picks sentences minimizing KL divergence w.r.t. document |
| **LUHN** | Uses frequency and sentence clusters to rank sentences |
| **SumBasic** | Probability-based sentence selection using word frequency |
| **Legal-SBERT** | Embedding-based semantic similarity for legal domain |

Each method generates an extractive summary that is later used in ETA assignment.

---

## 🧩 Extract-Then-Assign (ETA) — Core Idea

ETA transforms long documents into multiple aligned (chunk, summary) pairs.

### Steps:
1. Generate 7 extractive summaries for each document.  
2. Break the ground-truth summary into individual sentences.  
3. For each GT sentence, compute similarity with each extractive summary:  
   - ROUGE-1, ROUGE-2, ROUGE-L (average F1)  
4. Take **average similarity across all extractive sentences** → final score.  
5. Assign GT sentence to extractive summary with highest score.  
6. Apply length constraint:  
   > Assigned summary length must be ≤ 50% of extractive chunk length.  
   If exceeded → use 2nd best extractive chunk.  
7. Result: Up to 7 aligned training samples per document.

### Example:


GT Sentence: "Defendant’s motion was denied by the Court."
E2 Extractive: ["On March 2, the contract...", "Defendant’s motion denied..."]
Similarity ≈ 0.575  → highest score
✔ Assigned to E2



---

## 🧮 Training (BART + LoRA)

We fine-tune the BART-Large CNN model using LoRA to reduce memory usage.

### LoRA Configuration
```python
LoraConfig(
  r=16,
  lora_alpha=32,
  lora_dropout=0.1,
  target_modules=["q_proj", "v_proj"],
  task_type="SEQ_2_SEQ_LM"
)
````

### Training Arguments

```python
TrainingArguments(
  num_train_epochs=10,
  per_device_train_batch_size=4,
  gradient_accumulation_steps=2,
  learning_rate=1e-4,
  eval_strategy="epoch",
  save_strategy="epoch",
  fp16=True,
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss"
)
```

### Why LoRA?

* Trains only 5–10M parameters instead of 406M
* Saves >60% GPU memory
* Faster fine-tuning on free GPUs (Kaggle, Colab)

---

## 📊 Evaluation Metrics

We evaluate using:

### **ROUGE**

* ROUGE-1
* ROUGE-2
* ROUGE-L

### **BERTScore**

* Precision
* Recall
* F1

---

## 🏆 Results

```
ROUGE-1: 0.5723
ROUGE-2: 0.3547
ROUGE-L: 0.3169

BERTScore Precision: 0.8453
BERTScore Recall:    0.8535
BERTScore F1:        0.8493
```



## 🧭 Future Work

* Use long-context models (Longformer, LED, BigBird)
* Improve cross-chunk coherence with hierarchical summarization
* Add multilingual support for Indian courts
* Include citation reasoning and legal argument extraction


## ⭐ Acknowledgements

* Rishiai Dataset
* FIRE Legal Track
* HuggingFace Transformers
* PEFT (LoRA)
* ROUGE & BERTScore libraries
