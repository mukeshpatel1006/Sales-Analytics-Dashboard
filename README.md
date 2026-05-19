###
# AI Sales Prediction Dashboard

An end-to-end Machine Learning dashboard built using Python, Streamlit, and Scikit-learn.

## Features

- Sales prediction using Machine Learning
- Interactive Streamlit dashboard
- Real-time prediction
- Pretrained RandomForest model
- Clean UI
- Deployment ready

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```bash
ai-sales-dashboard/
│
├── app.py
├── sales_model.pkl
├── feature_columns.json
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Project

```bash
streamlit run app.py
```

## Model

The model is trained using RandomForestRegressor on Superstore sales dataset.

## Deployment

The dashboard can be deployed using Streamlit Community Cloud.
