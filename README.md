## PriceGemini 💎🔮✨

PriceGemini is a machine learning project to predict the price of diamonds, using a variety of features including carat, cut, color, clarity, and dimensions. The project uses an XGBoost model to achieve an R-squared score of 0.93 on the test set, which means that it can explain 93% of the variation in the target variable.

### How to Use

To use PriceGemini, simply provide the model with the following features:

* Carat ⚖️
* Cut 🔪
* Color 🎨
* Clarity 💎
* Depth 📐
* Table 📊
* X dimension 📏
* Y dimension 📏
* Z dimension 📏

The model will then output a prediction for the price of the diamond 💰

### Example

```python
import pricegemini

# Create a new PriceGemini model
model = pricegemini.PriceGeminiModel()

# Set the features for the diamond
model.carat = 1.0
model.cut = "Very Good"
model.color = "D"
model.clarity = "VS1"
model.depth = 60.0
model.table = 58.0
model.x_dimension = 5.0
model.y_dimension = 5.0
model.z_dimension = 3.0

# Predict the price of the diamond
predicted_price = model.predict_price()

# Print the predicted price
print(predicted_price)
```

Output:

```
$3263.0
```

### Benefits

PriceGemini can be used to benefit a variety of stakeholders, including:

* Diamond buyers 💎💍🛍️: PriceGemini can help diamond buyers to make more informed decisions about their purchases.
* Diamond sellers 💎💎💎: PriceGemini can help diamond sellers to set more competitive prices.
* Insurance companies 🛡️: PriceGemini can help insurance companies to more accurately value diamonds.

### Future Work

* Collect and use more data, especially data on higher-priced diamonds.
* Explore other machine learning algorithms and hyperparameter tuning techniques.
* Develop a web application or mobile app that allows users to predict the price of diamonds using the model.

### Contributing

If you are interested in contributing to the PriceGemini project, please feel free to submit a pull request.

### License

This project is licensed under the MIT License.
