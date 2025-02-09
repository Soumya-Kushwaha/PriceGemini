# PriceGemini: Diamond Price Prediction 💎

## Overview
PriceGemini is a machine learning project that predicts diamond prices based on various characteristics using regression analysis. The model leverages multiple features including the famous 4 Cs of diamonds (Cut, Color, Clarity, and Carat) along with other physical measurements to provide accurate price predictions.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Dataset Description
The dataset contains various attributes of diamonds along with their prices. 
Source: [Kaggle Playground Series S3E8](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

## Features

### The 4 Cs of Diamonds:
- **Carat** (0.2--5.01): Physical weight measured in metric carats
- **Cut** (Fair, Good, Very Good, Premium, Ideal): Quality of diamond cut
- **Color** (J to D, worst to best): Rating of diamond color
- **Clarity** (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF): Measure of diamond purity

### Physical Dimensions:
- **x**: Length in mm (0--10.74)
- **y**: Width in mm (0--58.9)
- **z**: Depth in mm (0--31.8)
- **depth**: Height from culet to table (%)
- **table**: Width of top facet relative to widest point (%)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PriceGemini.git
cd PriceGemini
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

The following preprocessing steps were implemented:
1. Removal of ID column
2. Handling of dimensionless diamonds (zeros in x, y, z)
3. Outlier removal based on regression analysis
4. Label encoding of categorical variables
5. Feature scaling using StandardScaler

## Model Development

### Pipeline Implementation:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Neighbors Regressor
- XGBoost Regressor

### Model Selection:
Models were evaluated using 10-fold cross-validation with negative root mean square error as the metric.

## Results

The XGBoost model achieved the best performance with:
- R² Score: ~0.98
- Adjusted R²: ~0.98
- Mean Absolute Error: ~328
- Root Mean Square Error: ~459

## Usage

```python
# Load the model and make predictions
import pandas as pd
from sklearn.pipeline import Pipeline

# Load your data
data = pd.read_csv('your_diamond_data.csv')

# Preprocess the data
# ... (preprocessing steps)

# Make predictions
predictions = model.predict(processed_data)
```

## Dependencies
```
numpy
pandas
seaborn
matplotlib
scikit-learn
xgboost
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by Kaggle
- Inspired by the need for accurate diamond price predictions in the gemstone industry
