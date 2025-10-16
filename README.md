# Loan Default Risk Prediction Using Bayesian Neural Network

A robust loan default prediction system using Bayesian Neural Networks (BNN) that provides both predictions and uncertainty quantification for better risk assessment in financial lending.

##  Project Overview

This project implements a Bayesian Neural Network to predict loan defaults while providing uncertainty estimates. Unlike traditional neural networks that give point predictions, BNNs provide probability distributions over predictions, enabling better risk management and decision-making in loan approval processes.

##  Key Features

- **Uncertainty Quantification**: Model provides confidence intervals for each prediction
- **Robust Predictions**: Bayesian approach handles model uncertainty naturally
- **Risk Assessment**: Different confidence levels for conservative vs aggressive lending strategies
- **Comprehensive Evaluation**: Multiple metrics and visualizations for model assessment

##  Model Performance

### Overall Metrics
- **ROC AUC**: 0.97 (Excellent discrimination)
- **Average Precision**: 0.91 (Outstanding precision-recall performance)
- **Model Architecture**: 2-layer BNN with 20 hidden units per layer

### Performance Summary
```
                    Precision    Recall    F1-Score    Support
    No Default         0.95      0.98      0.96       1600
    Default            0.94      0.87      0.90        400
    
    Accuracy: 0.95
    Macro Avg: 0.94
    Weighted Avg: 0.95
```

## 🏦 Business Applications

### Three-Tier Decision Framework

| Risk Level | Default Probability | Action | Confidence |
|------------|-------------------|---------|------------|
| **Low Risk** | < 15% | Auto-Approve  | High |
| **Medium Risk** | 15-35% | Manual Review  | Variable |
| **High Risk** | > 35% | Auto-Reject  | High |

### Sample Predictions

#### Conservative Applicant (APPROVED)
```
Profile: Age 35, Bachelor's, $75K income, Owns home
Loan: $15K for home improvement, 8.5% rate, 20% debt ratio
Result: 2.09% default risk, HIGH confidence → APPROVE
```

#### Borderline Applicant (MANUAL REVIEW)
```
Profile: Age 42, Bachelor's, $68K income, Owns home  
Loan: $28K personal loan, 12.8% rate, 41% debt ratio
Result: 23.96% default risk, LOW confidence → MANUAL REVIEW
```

#### High-Risk Applicant (REJECTED)
```
Profile: Age 28, Master's, $55K income, Rents
Loan: $35K debt consolidation, 18.75% rate, 64% debt ratio
Result: 99.91% default risk, HIGH confidence → REJECT
```

## 🛠️ Technical Implementation

### Dependencies
```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
jax>=0.3.0
numpyro>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Model Architecture
- **Input Layer**: Preprocessed features (numerical + categorical)
- **Hidden Layer 1**: 20 units with tanh activation
- **Hidden Layer 2**: 20 units with tanh activation  
- **Output Layer**: 1 unit with sigmoid activation
- **Prior**: Normal(0, 0.5) for all weights and biases
- **Inference**: NUTS sampler with 1000 warmup + 2000 samples

### Data Preprocessing
1. **Numerical Features**: Standardized using StandardScaler
2. **Categorical Features**: One-hot encoded with drop_first=True
3. **Missing Values**: Filled with median (numerical) or "missing" (categorical)
4. **Sampling**: Balanced training set with 10,000 samples
5. **Train/Val Split**: 80/20 with stratification

##  Model Evaluation

### ROC Curve Analysis
- **AUC = 0.97**: Exceptional discrimination between classes
- **High TPR at Low FPR**: Can catch 90% of defaults with only 5-10% false positives
- **Business Impact**: Excellent for both conservative and aggressive lending strategies

### Precision-Recall Analysis  
- **AP = 0.91**: Outstanding precision across all recall levels
- **High Precision Plateau**: Nearly 100% precision at low recall levels
- **Flexible Thresholding**: Supports different business risk tolerances

### Uncertainty Distribution
- **Bimodal Pattern**: Clear separation between confident predictions
- **Rare Uncertainty**: Few borderline cases require manual review
- **Meaningful Doubt**: When model is uncertain, it's genuinely difficult cases

##  Usage

### Training the Model
```python
# Load and preprocess data
X_train, X_val, y_train, y_val = preprocess_data(X_df, y, sample_size=10000)

# Train BNN model
samples = run_hmc(X_train, y_train, D_H=20, num_warmup=1000, num_samples=2000)

# Make predictions with uncertainty
probs, uncertainties = predict(samples, X_val)
```

### Making Predictions
```python
# Preprocess new applicant
new_sample = preprocess_new_sample(applicant_data, X_df, training_features, 
                                   training_scaler, training_medians)

# Get prediction with uncertainty
prob, uncertainty = predict(samples, new_sample)

# Business decision
if uncertainty < 0.05:
    confidence = "HIGH"
    decision = "AUTO_APPROVE" if prob < 0.15 else "AUTO_REJECT"
else:
    confidence = "LOW" 
    decision = "MANUAL_REVIEW"
```

##  Project Structure
```
├── data/
│   └── loan_data.csv           # Loan dataset
├── BNN.ipynb                   # Main implementation notebook

```

##  Key Insights

1. **Debt-to-Income Ratio**: Strongest predictor of default risk
2. **Employment History**: Longer tenure reduces default probability  
3. **Home Ownership**: Provides collateral security, reduces risk
4. **Credit Score**: Important but not deterministic alone
5. **Loan Purpose**: Debt consolidation higher risk than home improvement

---

##  References

- [Bayesian Neural Networks in NumPyro](https://num.pyro.ai/)
- [NUTS Sampler Documentation](https://mc-stan.org/docs/2_18/reference-manual/hmc-chapter.html)
- [Loan Default Prediction Best Practices](https://www.federalreserve.gov/publications/files/risk-management-at-banks-201612.pdf)
---

**🎯 Built with Bayesian Neural Networks for robust loan default prediction with uncertainty quantification.**
