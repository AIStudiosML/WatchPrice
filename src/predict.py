import os
import numpy as np
import config
import pandas as pd
from train import TrainModel
from data import PreprocessData


class Product:
    def __init__(self, brand, current_price, original_price, discount_percentage, rating, num_ratings, model_name, dial_shape, strap_color, strap_material, touchscreen, battery_life_days, bluetooth, display_size, weight):
        self.brand = brand
        self.current_price = current_price
        self.original_price = original_price
        self.discount_percentage = discount_percentage
        self.rating = rating
        self.num_ratings = num_ratings
        self.model_name = model_name
        self.dial_shape = dial_shape
        self.strap_color = strap_color
        self.strap_material = strap_material
        self.touchscreen = touchscreen
        self.battery_life_days = battery_life_days
        self.bluetooth = bluetooth
        self.display_size = display_size
        self.weight = weight

    def to_dataframe(self):
        data = {
            'Unnamed: 0': 0,
            'Brand': [self.brand or np.nan],
            'Current Price': [self.current_price or np.nan],
            'Original Price': [self.original_price or np.nan],
            'Discount Percentage': [self.discount_percentage or np.nan],
            'Rating': [self.rating or np.nan],
            'Number OF Ratings': [self.num_ratings or np.nan],
            'Model Name': [self.model_name or np.nan],
            'Dial Shape': [self.dial_shape or np.nan],
            'Strap Color': [self.strap_color or np.nan],
            'Strap Material': [self.strap_material or np.nan],
            'Touchscreen': [self.touchscreen or np.nan],
            'Battery Life (Days)': [self.battery_life_days or np.nan],
            'Bluetooth': [self.bluetooth or np.nan],
            'Display Size': [self.display_size or np.nan],
            'Weight': [self.weight or np.nan]
        }
        return pd.DataFrame.from_dict(data)


class Predict:
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self):
        tm = TrainModel()
        self.model = tm.load_model()

    def predict_row(self, row: Product):
        self.df = row.to_dataframe()
        ps = PreprocessData()
        self.df = ps.clean_df_predict(self.df)
        return self.df


# print(os.getcwd())
# p = Predict()
# d = p.predict_row(Product('noise', 82990, 89900, 7.686318131, 4, 65,
#                   'BSW046', None, None, None, None, '8', 'Yes', None, '35 - 50 g'))

# tm = TrainModel.load_model()
# pred = tm.predict(d.drop(['Discount Price'], axis=1))
# print(pred)
