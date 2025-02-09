# Diamond Price Predictor 💎

A machine learning project to predict diamond prices based on various characteristics using multiple regression models.

## Project Overview

This project analyzes a diamond dataset to predict prices based on features like carat weight, cut quality, color, clarity, and physical dimensions. It implements several machine learning models including XGBoost, Random Forest, and Linear Regression to find the best price prediction model.

## Dataset Description

The dataset includes the following features:
- `carat`: Weight of the diamond (0.2--5.01)
- `cut`: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- `color`: Diamond color, from J (worst) to D (best)
- `clarity`: Clarity grade (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
- `depth`: Height from culet to table (54-70)
- `table`: Width of top facet relative to widest point (50-73)
- `x`: Length (mm)
- `y`: Width (mm)
- `z`: Depth (mm)
- `price`: Price in USD (target variable)

## Project Structure
```
diamond-price-predictor/
├── data/
│   └── gemstone.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/diamond-price-predictor.git
cd diamond-price-predictor
pip install -r requirements.txt
```

## Requirements
```
numpy
pandas
seaborn
matplotlib
scikit-learn
xgboost
```

## Usage

1. Data Preprocessing:
```python
python src/data_preprocessing.py
```

2. Model Training:
```python
python src/model_training.py
```

3. Evaluation:
```python
python src/evaluation.py
```

## Model Performance

The XGBoost model achieved the best performance with:
- R²: 0.98
- Adjusted R²: 0.98
- MAE: 368.72
- RMSE: 539.42

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.