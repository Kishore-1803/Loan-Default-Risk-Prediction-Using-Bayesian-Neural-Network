```markdown
# 🏦 Loan Default Risk Prediction Using Bayesian Neural Networks

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPyro](https://img.shields.io/badge/NumPyro-Bayesian%20Inference-orange.svg)](https://num.pyro.ai/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated%20ML-red.svg)](https://jax.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)](#)

A **production-ready loan default prediction system** built using **Bayesian Neural Networks (BNNs)**.  
Unlike traditional models that produce only point estimates, this system provides **probabilistic predictions with uncertainty quantification**, enabling **risk-aware and explainable lending decisions**.

---

## 📌 Project Overview

Financial lending decisions involve inherent uncertainty. This project leverages **Bayesian Neural Networks** to:

- Predict loan default probabilities  
- Quantify predictive uncertainty  
- Enable **confidence-driven decision-making** (auto-approve, manual review, auto-reject)

The system is designed for **banks, NBFCs, and fintech platforms** that require **robust, transparent, and risk-sensitive credit assessment**.

---

## ✨ Key Features

- **Uncertainty Quantification**  
  Confidence intervals for every prediction using Bayesian inference  

- **Robust Risk Modeling**  
  Naturally accounts for model and data uncertainty  

- **Decision-Oriented Outputs**  
  Supports conservative, balanced, and aggressive lending strategies  

- **Comprehensive Evaluation**  
  ROC, Precision-Recall, classification metrics, and uncertainty analysis  

---

## 📊 Model Performance

### Overall Metrics

| Metric | Score |
|------|------|
| **ROC AUC** | **0.97** |
| **Average Precision (AP)** | **0.91** |
| **Accuracy** | **0.95** |
| **Architecture** | 2-layer BNN (20 hidden units per layer) |

### Classification Report

```

```
                Precision    Recall    F1-Score    Support
```

No Default             0.95       0.98       0.96        1600
Default                0.94       0.87       0.90         400

Accuracy: 0.95
Macro Avg: 0.94
Weighted Avg: 0.95

```

---

## 🧠 Business Decision Framework

### Three-Tier Risk-Based Strategy

| Risk Level | Default Probability | Action | Confidence |
|-----------|-------------------|--------|------------|
| **Low Risk** | < 15% | Auto-Approve | High |
| **Medium Risk** | 15–35% | Manual Review | Variable |
| **High Risk** | > 35% | Auto-Reject | High |

---

## 🧾 Sample Predictions

### ✅ Conservative Applicant — *APPROVED*
```

Age: 35 | Education: Bachelor’s | Income: $75K | Home: Owned
Loan: $15K (Home Improvement), 8.5% interest, 20% DTI
Prediction: 2.09% default risk
Confidence: HIGH → AUTO APPROVE

```

### ⚠️ Borderline Applicant — *MANUAL REVIEW*
```

Age: 42 | Education: Bachelor’s | Income: $68K | Home: Owned
Loan: $28K (Personal), 12.8% interest, 41% DTI
Prediction: 23.96% default risk
Confidence: LOW → MANUAL REVIEW

```

### ❌ High-Risk Applicant — *REJECTED*
```

Age: 28 | Education: Master’s | Income: $55K | Home: Rented
Loan: $35K (Debt Consolidation), 18.75% interest, 64% DTI
Prediction: 99.91% default risk
Confidence: HIGH → AUTO REJECT

````

---

## ⚙️ Technical Implementation

### Tech Stack

- **Python**
- **NumPy, Pandas**
- **Scikit-learn**
- **JAX**
- **NumPyro**
- **Matplotlib & Seaborn**

### Model Architecture

- **Input Layer**: Preprocessed numerical & categorical features  
- **Hidden Layer 1**: 20 units (tanh)  
- **Hidden Layer 2**: 20 units (tanh)  
- **Output Layer**: Sigmoid (default probability)  
- **Prior**: Normal(0, 0.5)  
- **Inference**: NUTS (HMC)  
  - 1000 warmup steps  
  - 2000 posterior samples  

---

## 🔄 Data Preprocessing Pipeline

1. Numerical features → **StandardScaler**
2. Categorical features → **One-hot encoding**
3. Missing values →  
   - Numerical: Median  
   - Categorical: `"missing"`
4. Balanced sampling → **10,000 records**
5. Train-validation split → **80/20 (stratified)**

---

## 📈 Model Evaluation Highlights

### ROC Analysis
- Excellent separation between defaulters and non-defaulters  
- Captures ~90% of defaults at low false-positive rates  

### Precision-Recall
- High precision maintained across recall levels  
- Ideal for flexible risk thresholding  

### Uncertainty Insights
- Clear bimodal confidence distribution  
- Uncertainty aligns with genuinely ambiguous cases  
- Perfect for triggering **manual review workflows**

---

## 🚀 Usage

### Train the Model
```python
X_train, X_val, y_train, y_val = preprocess_data(
    X_df, y, sample_size=10000
)

samples = run_hmc(
    X_train, y_train,
    D_H=20,
    num_warmup=1000,
    num_samples=2000
)
````

### Predict with Uncertainty

```python
probs, uncertainties = predict(samples, X_val)
```

### Business Decision Logic

```python
if uncertainty < 0.05:
    decision = "AUTO_APPROVE" if prob < 0.15 else "AUTO_REJECT"
else:
    decision = "MANUAL_REVIEW"
```

---

## 📁 Project Structure

```
├── data/
│   └── loan_data.csv
├── BNN.ipynb
└── README.md
```

---

## 🔍 Key Insights

* **Debt-to-Income Ratio** is the strongest predictor
* **Employment stability** reduces default risk
* **Home ownership** provides risk mitigation
* **Credit score alone is insufficient**
* **Debt consolidation loans carry higher risk**

---

## 📚 References

* Bayesian Neural Networks in NumPyro
* NUTS / Hamiltonian Monte Carlo
* Federal Reserve – Risk Management in Banking

---

## 👥 Contributors

* **Kishore B** – [GitHub](https://github.com/Kishore-1803)
* **Naveen Babu M S** – [GitHub](https://github.com/naveen-astra)
* **Koushal Reddy M** – [GitHub](https://github.com/mendu645)
* **Sai Charan M**

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

⭐ *If you find this project useful, consider giving it a star!* ⭐

```
```
