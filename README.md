# 📦 Demand Forecasting with XGBoost

An end-to-end **machine learning pipeline** for predicting product demand across retail stores using **XGBoost Regression**, complete with exploratory data analysis, hyperparameter tuning, and an interactive **Streamlit** web application for real-time predictions.

---

## 🎯 Project Overview

This project tackles the challenge of forecasting product demand in a multi-store, multi-category retail environment. Accurate demand forecasting helps businesses optimize inventory management, reduce waste, and improve supply chain efficiency.

The pipeline covers:
- **Exploratory Data Analysis (EDA)** — Uncovering patterns, distributions, and relationships in the data
- **Feature Engineering** — Creating time-based and derived features to improve model performance
- **Model Training** — Building an XGBoost regressor with `RandomizedSearchCV` hyperparameter tuning
- **Deployment** — Serving predictions through an interactive Streamlit dashboard

---

## 📂 Project Structure

```
DemandForcasting_Project/
│
├── DemandForcasting.ipynb          # EDA & data preprocessing notebook
├── machine_learning.ipynb          # Model training, tuning & evaluation notebook
├── app.py                          # Streamlit web app for real-time predictions
│
├── demand_forecasting.csv          # Raw dataset (76,000 records)
├── Preprocessed_demand_forecasting_data.csv  # Cleaned & feature-engineered dataset
│
├── xgboost_demand.pkl              # Serialized XGBoost model (initial)
├── xgboost_demand_model.pkl        # Serialized XGBoost model (tuned, best)
├── label_encoders.pkl              # Serialized label encoders for categorical features
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset contains **76,000 records** spanning from **January 2022 to January 2024**, tracking sales and demand across **5 stores**, **20 products**, and **5 categories**.

| Feature | Description | Type |
|---|---|---|
| `Date` | Transaction date | datetime |
| `Store ID` | Store identifier (S001–S005) | categorical |
| `Product ID` | Product identifier (P0001–P0020) | categorical |
| `Category` | Product category (Electronics, Clothing, Groceries, Furniture, Toys) | categorical |
| `Region` | Store region (North, South, East, West) | categorical |
| `Inventory Level` | Current stock level | numeric |
| `Units Sold` | Number of units sold | numeric |
| `Units Ordered` | Number of units ordered | numeric |
| `Price` | Product price ($4.74 – $228.03) | numeric |
| `Discount` | Discount percentage (0% – 25%) | numeric |
| `Weather Condition` | Weather at time of sale (Sunny, Cloudy, Rainy, Snowy) | categorical |
| `Promotion` | Whether a promotion was active (0/1) | binary |
| `Competitor Pricing` | Competitor's price for similar product | numeric |
| `Seasonality` | Season (Spring, Summer, Fall, Winter) | categorical |
| `Epidemic` | Whether an epidemic was ongoing (0/1) | binary |
| **`Demand`** | **Target variable — units demanded** | **numeric** |

> **Data Quality:** The dataset is clean — **zero missing values** and **zero duplicate rows**.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | XGBoost, Scikit-learn |
| Hyperparameter Tuning | `RandomizedSearchCV` (3-fold CV, 25 iterations) |
| Model Serialization | Pickle |
| Web App | Streamlit |

---

## 🤖 Model Details

### Algorithm
**XGBoost Regressor** (`XGBRegressor`) with `reg:squarederror` objective.

### Features Used
The model uses the following **6 features** to predict demand:

| Feature | Importance |
|---|---|
| Category | 38.2% |
| Promotion | 36.5% |
| Price | 11.2% |
| Discount | 6.2% |
| Competitor Pricing | 4.6% |
| Inventory Level | 3.3% |

### Best Hyperparameters (via RandomizedSearchCV)

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 1 |

### Performance
| Metric | Score |
|---|---|
| **RMSE** | **35.42** |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shivamGupta-25/Demand-Forecasting.git
   cd Demand-Forecasting
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
   ```

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501` and provides an interactive interface to:
- Input product **Price**, **Discount**, **Inventory Level**, **Promotion** status, **Competitor Price**, and **Category**
- Get an instant **demand prediction** in units

---

## 📓 Notebooks

### 1. `DemandForcasting.ipynb` — EDA & Preprocessing
- Data loading, type conversion, and quality checks
- Statistical summaries and distribution analysis
- Feature engineering: `Year`, `Month`, `Day`, `Weekday`, `Discounted Price`
- Visualizations: correlation heatmaps, category distributions, seasonal trends, and more
- Exports cleaned data to `Preprocessed_demand_forecasting_data.csv`

### 2. `machine_learning.ipynb` — Model Training & Evaluation
- Feature selection (6 key predictors)
- Label encoding for categorical variables
- 80/20 train-test split
- XGBoost model with `RandomizedSearchCV` (25 iterations × 3-fold CV = 75 fits)
- Feature importance analysis and visualization
- Model and encoder serialization to `.pkl` files

---

## 📝 License

This project is open-source and available for educational and research purposes.

---

## 🙋 Author

**Shivam Raj Gupta**

---

> Built with ❤️ using Python, XGBoost, and Streamlit
