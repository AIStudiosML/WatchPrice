import os
import config
import pickle
import pandas as pd
import xgboost as xgb
from data import PreprocessData
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


class TrainModel:
    def __init__(self):
        self.df = self._read_df()

    def _read_df(self):
        try:
            if not os.path.exists(config.CLEAN_FILE_PATH):
                raise FileNotFoundError(
                    f'file: {config.CLEAN_FILE_PATH} not fount!')
        except Exception as e:
            print(e)
            pdo = PreprocessData()
            pdo.clean_df()
        return pd.read_csv(config.CLEAN_FILE_PATH)

    def _split_data(self):
        self.X = self.df.drop('Discount Price', axis=1)
        self.y = self.df['Discount Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)  # 80% training, 20% testing

    def _train_decision_tree(self):
        dt_model = DecisionTreeRegressor()
        dt_model.fit(self.X_train, self.y_train)
        y_pred = dt_model.predict(self.X_test)
        score = cross_val_score(dt_model, self.X, self.y, cv=5, scoring='r2')
        return {"score": score, "mean": score.mean(), "std": score.std(), "model": dt_model}

    def _train_random_forest(self):
        rf_model = RandomForestRegressor()
        rf_model.fit(self.X_train, self.y_train)
        y_pred = rf_model.predict(self.X_test)
        score = cross_val_score(rf_model, self.X, self.y, cv=5, scoring='r2')
        return {"score": score, "mean": score.mean(), "std": score.std(), "model": rf_model}

    def _train_xgb(self):
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(self.X_train, self.y_train)
        y_pred = xgb_model.predict(self.X_test)
        score = cross_val_score(xgb_model, self.X, self.y, cv=5, scoring='r2')
        return {"score": score, "mean": score.mean(), "std": score.std(), "model": xgb_model}

    def _hyperparameter_train_xgb(self):
        model = xgb.XGBRegressor()
        param_gird = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        grid_search = GridSearchCV(
            estimator=model, param_grid=param_gird, cv=5, scoring='r2')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_

        best_model = xgb.XGBRegressor(**best_params)

        best_model.fit(self.X_train, self.y_train)
        y_pred = best_model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        return r2, best_params, best_model

    def _save_model(self, model):
        with open(config.MODEL_SAVE_NAME, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model():
        with open(config.MODEL_SAVE_NAME, 'rb') as f:
            return pickle.load(f)

    def train(self):
        print('Training started!')
        self._split_data()
        print('Data split complete!')
        dt_result = self._train_decision_tree()
        print('Decision tree training completed!', dt_result)
        rf_result = self._train_random_forest()
        print('Random Forest training completed!', rf_result)
        xgb_result = self._train_xgb()
        print('XGB training completed!', xgb_result)
        print('XGB Hyperparameter started!')
        r2, best_params, model = self._hyperparameter_train_xgb()
        print('XGB Hyperparameter End!')
        self._save_model(model)
        return {"dt_result": dt_result, "rf_result": rf_result, "xgb_result": xgb_result, "best_r2": r2, "best_xgb_param": best_params}
