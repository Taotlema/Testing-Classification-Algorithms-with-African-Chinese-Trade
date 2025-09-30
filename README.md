# Predicting-China-Africa-Trade-Shipment-Delays

International trade profoundly impacts communities, shaping what kinds of work, capital, and goods are accessible. The growing trade relationship between China and Africa presents unique opportunities for development, but supply chain disruptions—particularly shipment delays—can create significant challenges. These disruptions lead to financial losses, inventory shortages, and difficulties in planning for businesses and communities that rely on these flows.

This project investigates a critical question:  
**Can we predict shipment delays in the China–Africa trade corridor using trade characteristics such as transport mode, commodity type, shipment value, and route information?**

This is framed as a **binary classification problem**:  
- **0 = On-Time**  
- **1 = Delayed**  

---

## Introduction

By identifying patterns that signal potential delays, stakeholders could proactively manage risks, optimize logistics planning, and minimize disruptions. My investigation applies machine learning methods to a simulated trade monitoring dataset representing 10,000 China–Africa shipment transactions.

The overarching goal is to understand whether **logistics and financial features alone** can predict shipment delays—or whether other hidden factors (e.g., weather, port congestion, inspections) are more decisive.

---

## Understanding the Data

The dataset is from Kaggle:  
[China-Africa Trade Monitoring Dataset](https://www.kaggle.com/datasets/ziya07/chinaafrica-trade-monitoring-dataset)

### Key Features
- **Shipment Details:** `shipment_id`, `export_country`, `import_country`, `departure_port`, `arrival_port`  
- **Commodities:** `commodity`, `hs_code`, `quantity`  
- **Financials:** `declared_value_usd`, `contract_value_usd`, `currency_exchange_rate`, `market_price_per_unit`  
- **Logistics:** `transport_mode`, `transit_time_days`, `payment_terms`  
- **Targets:** `delay_status` (categorical), later converted into `delay_binary`  

### Class Balance
- **On-Time:** ~79%  
- **Delayed:** ~21%  
- → Dataset is **imbalanced**, which complicates model performance.

---

## Data Preparation

### Preprocessing Steps
1. **Validation:** No duplicates, null values, or incorrect datatypes.  
2. **Target Creation:** Converted `delay_status` → binary `delay_binary`.  
3. **Feature Selection:** Dropped identifiers (`shipment_id`), redundant labels (`delay_status`), and anomaly flags.  
4. **Categorical Encoding:** Applied one-hot encoding with `drop_first=True`. Expanded from **14 to 33 features**.  
5. **Train/Test Split:** 70% train, 30% test (stratified to preserve class balance).  

---

## Exploratory Analysis

### Delay Distribution
The dataset shows a **3.81:1 ratio** of on-time to delayed shipments.  
![Delay Distribution](delay_distribution.png)

### Transport Mode
- Road and ship handle most shipments.  
- Air and rail are smaller contributors, but all modes experience some delays.  

### Commodity Types
- Electronics, Machinery, Textiles, Agricultural, and Minerals appear evenly distributed between delayed and on-time categories.  

### Transit Time
- Longer planned duration does not reliably predict delays.  

### Price Volatility
- Delayed shipments show slightly higher volatility on average, suggesting some correlation.  

---

## Modeling Approach

### Algorithm: Random Forest Classifier
Random Forest was selected for its ability to:  
- Handle mixed categorical + numeric data  
- Capture non-linear patterns  
- Manage class imbalance via `class_weight='balanced'`  
- Provide **feature importance metrics**  

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced'
)
```

---

## Evaluation

### Cross-Validation (10-Fold)
- Accuracy: **78.20% ± 0.46%**  
- Precision: **23.28% ± 9.94%**  
- Recall: **2.12% ± 1.08%**  
- F1-Score: **3.86% ± 1.90%**  

### Test Set Results
- Accuracy: **78.37%**  
- ROC-AUC: **0.52**  

### Confusion Matrix
- True Negatives: 2,338  
- False Positives: 38  
- False Negatives: 611  
- True Positives: 13  

**Takeaway:**  
Despite high accuracy, the model **missed 98% of delays**. The imbalance caused accuracy to be misleading.

---

## Interpretation

### Insights
- Financial indicators (`declared_value_usd`, `contract_value_usd`) were stronger predictors than logistics (`transit_time_days`, `transport_mode`).  
- Traditional shipment details (commodity type, route) offered little predictive power.  

### Missing Factors
- Weather conditions  
- Port congestion and strikes  
- Customs inspections  
- Political instability  

These missing features likely explain the model’s weak recall.

---

## Conclusion and Takeaways

**Key Finding:** Current dataset features are inadequate for predicting shipment delays.  

- The Random Forest model achieved **recall of 2%**, indicating almost all delays were missed.  
- **ROC-AUC of 0.52** is close to random guessing.  

**Takeaway:** Shipment delays require **contextual external data sources** (e.g., real-time weather, port congestion) for meaningful prediction.

---

## Impact

### Potential Benefits (with improved data)
- Better inventory and cash flow planning  
- Optimized logistics routing  
- Reduced financial losses  
- Improved customer reliability  

### Risks
1. **False Confidence:** Businesses relying on flawed predictions may suffer losses.  
2. **Bias:** Model may unfairly prioritize high-value shipments.  
3. **Overreliance:** Machine learning solutions might overshadow human expertise.  
4. **Dataset Limits:** Simulated nature reduces real-world reliability.  

---

## Ethical Considerations
- **Transparency:** Clearly communicate model limitations.  
- **Accountability:** Define responsibility when predictions fail.  
- **Equity:** Ensure small businesses are not disadvantaged.  
- **Autonomy:** Keep humans in the decision loop.  

---

## References
1) Kaggle. (2024). *China-Africa Trade Monitoring Dataset*. Retrieved from [Kaggle](https://www.kaggle.com/datasets/ziya07/chinaafrica-trade-monitoring-dataset)  
2) Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.  
3) Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.  
4) Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. J. of AI Research, 16, 321-357.  
5) Python Software Foundation. (2023). *Python 3.x*. Retrieved from [python.org](https://www.python.org/)  
6) McKinney, W. (2023). *pandas: Python Data Analysis Library*. Retrieved from [pandas.pydata.org](https://pandas.pydata.org/)  
7) Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. Computing in Sci & Eng, 9(3), 90-95.  
8) Waskom, M. L. (2021). *seaborn: statistical data visualization*. JOSS, 6(60), 3021.  

---

*Generative AI (Claude Sonnet 4.5) was leveraged when learning to implement classification algorithms and optimize code for this project.*
