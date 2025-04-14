# Used Car Price Prediction

This project predicts used car prices using machine learning techniques. The system processes raw car listing data, applies preprocessing steps, trains a predictive model, and provides functionality to make price predictions.

## Repository Structure


## Structure
```
└── sakshamtapadia-used_car_price_pred./
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   └── raw/
    │       ├── used_car.csv
    │       └── used_cars2.csv
    ├── models/
    │   └── saved_models/
    │       ├── model.pkl
    │       └── preprocessors.pkl
    ├── notebooks/
    │   └── exploration.ipynb
    └── src/
        ├── config/
        │   └── config.py
        ├── data_processing/
        │   └── preprocessing.py
        ├── models/
        │   ├── predict.py
        │   └── train.py
        ├── tests/
        │   └── test_preprocessing.py
        └── utils/
            └── helpers.py

```

## Features

- Data preprocessing pipeline for cleaning and transforming car listing data
- Machine learning model for price prediction
- Saved models for easy deployment
- Comprehensive data exploration notebooks
- Modular code structure for maintainability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sakshamtapadia/used_car_price_pred.git
cd used_car_price_pred
```
## How to Run

```bash
pip install -r requirements.txt
python src/models/train.py

Making Predictions
Use the prediction script (implementation details in src/models/predict.py).

Data Exploration
Open and run the Jupyter notebook in notebooks/exploration.ipynb.

Data
The dataset contains the following columns:

Brand: Manufacturer of the car

model: Specific model name

Year: Manufacturing year

Age: Age of the car

kmDriven: Kilometers driven

Transmission: Type of transmission

Owner: Number of previous owners

FuelType: Type of fuel used

PostedDate: When the listing was posted

AdditionInfo: Additional information

AskPrice: Target variable - asking price

Dependencies
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

Contributing
  Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
MIT License (Note: You may want to add a proper license file)