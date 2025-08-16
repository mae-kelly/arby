import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import joblib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, task_type='regression', random_state=42):
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = None
        self.cv_results = {}
        
    def get_model_configs(self):
        if self.task_type == 'regression':
            return {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 6, 9],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 6, 9],
                        'num_leaves': [31, 50, 100]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 6, 9]
                    }
                },
                'ridge': {
                    'model': Ridge(random_state=self.random_state),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                },
                'lasso': {
                    'model': Lasso(random_state=self.random_state),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                },
                'elastic_net': {
                    'model': ElasticNet(random_state=self.random_state),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0],
                        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                    }
                },
                'svr': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto'],
                        'epsilon': [0.01, 0.1, 0.2]
                    }
                }
            }
    
    def train_single_model(self, model_name, X_train, y_train, X_val=None, y_val=None, 
                          optimize_hyperparams=True, cv_folds=5):
        model_config = self.get_model_configs()[model_name]
        model = model_config['model']
        param_grid = model_config['params']
        
        if optimize_hyperparams and len(param_grid) > 0:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            self.models[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_score': -grid_search.best_score_
            }
        else:
            model.fit(X_train, y_train)
            
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_score = mean_squared_error(y_val, val_pred)
            else:
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                          scoring='neg_mean_squared_error')
                val_score = -cv_scores.mean()
            
            self.models[model_name] = {
                'model': model,
                'best_params': {},
                'cv_score': val_score
            }
        
        return self.models[model_name]
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, 
                        model_names=None, optimize_hyperparams=True):
        if model_names is None:
            model_names = list(self.get_model_configs().keys())
        
        for model_name in model_names:
            print(f"Training {model_name}...")
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, optimize_hyperparams
                )
                print(f"{model_name} CV Score: {result['cv_score']:.4f}")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        self.select_best_model()
        return self.models
    
    def select_best_model(self):
        if not self.models:
            return None
        
        best_model_name = min(self.models.keys(), 
                             key=lambda x: self.models[x]['cv_score'])
        
        self.best_model = self.models[best_model_name]['model']
        self.best_score = self.models[best_model_name]['cv_score']
        
        print(f"Best model: {best_model_name} with CV score: {self.best_score:.4f}")
        return best_model_name
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        if len(y_test) > 1:
            directional_accuracy = np.mean(
                np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred))
            )
            metrics['directional_accuracy'] = directional_accuracy
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            metrics = self.evaluate_model(model, X_test, y_test)
            results[model_name] = metrics
        
        return results
    
    def predict(self, X, model_name=None):
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Train models first.")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.models[model_name]['model']
        
        return model.predict(X)
    
    def save_models(self, filepath):
        save_data = {
            'models': self.models,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'task_type': self.task_type,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_models(self, filepath):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.best_model = save_data['best_model']
        self.best_score = save_data['best_score']
        self.task_type = save_data['task_type']
        self.random_state = save_data['random_state']

class NeuralNetworkTrainer:
    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=1, dropout=0.2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def create_model(self):
        layers = []
        
        prev_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.model = nn.Sequential(*layers).to(self.device)
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, 
              batch_size=32, learning_rate=0.001, patience=10):
        
        if self.model is None:
            self.create_model()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_nn_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")
                if X_val is not None:
                    print(f"Val Loss: {val_loss:.4f}")
        
        if X_val is not None and y_val is not None:
            self.model.load_state_dict(torch.load('best_nn_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout': self.dropout
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_size = checkpoint['input_size']
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.output_size = checkpoint['output_size']
        self.dropout = checkpoint['dropout']
        
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

class HyperparameterOptimizer:
    def __init__(self, model_type, task_type='regression'):
        self.model_type = model_type
        self.task_type = task_type
        
    def objective(self, trial, X_train, y_train, cv_folds=5):
        if self.model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            model = xgb.XGBRegressor(**params, random_state=42)
            
        elif self.model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            model = lgb.LGBMRegressor(**params, random_state=42)
            
        elif self.model_type == 'neural_network':
            hidden_size_1 = trial.suggest_int('hidden_size_1', 32, 256)
            hidden_size_2 = trial.suggest_int('hidden_size_2', 16, 128)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
            
            nn_trainer = NeuralNetworkTrainer(
                input_size=X_train.shape[1],
                hidden_sizes=[hidden_size_1, hidden_size_2],
                dropout=dropout
            )
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                nn_trainer.create_model()
                nn_trainer.train(
                    X_fold_train, y_fold_train, 
                    X_fold_val, y_fold_val,
                    epochs=50, learning_rate=learning_rate
                )
                
                y_pred = nn_trainer.predict(X_fold_val)
                score = mean_squared_error(y_fold_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                               scoring='neg_mean_squared_error')
        
        return -scores.mean()
    
    def optimize(self, X_train, y_train, n_trials=100, cv_folds=5):
        study = optuna.create_study(direction='minimize')
        
        objective_with_data = lambda trial: self.objective(
            trial, X_train, y_train, cv_folds
        )
        
        study.optimize(objective_with_data, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial
        }

class EnsembleTrainer:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or LinearRegression()
        self.fitted = False
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        tscv = TimeSeriesSplit(n_splits=5)
        
        meta_features_train = np.zeros((X_train.shape[0], len(self.base_models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            for model_idx, (name, model) in enumerate(self.base_models.items()):
                model_copy = self._clone_model(model)
                model_copy.fit(X_fold_train, y_fold_train)
                
                val_pred = model_copy.predict(X_fold_val)
                meta_features_train[val_idx, model_idx] = val_pred
        
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
        
        self.meta_model.fit(meta_features_train, y_train)
        self.fitted = True
        
        return self
    
    def _clone_model(self, model):
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            return type(model)(**model.get_params())
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            pred = model.predict(X)
            meta_features[:, model_idx] = pred
        
        return self.meta_model.predict(meta_features)
    
    def get_base_predictions(self, X):
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        return predictions

class ModelValidation:
    def __init__(self, validation_method='time_series'):
        self.validation_method = validation_method
        
    def time_series_validation(self, model, X, y, n_splits=5, test_size=0.2):
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * test_size))
        
        scores = {
            'mse': [],
            'mae': [],
            'r2': [],
            'directional_accuracy': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            
            y_pred = model_copy.predict(X_test)
            
            scores['mse'].append(mean_squared_error(y_test, y_pred))
            scores['mae'].append(mean_absolute_error(y_test, y_pred))
            scores['r2'].append(r2_score(y_test, y_pred))
            
            if len(y_test) > 1:
                dir_acc = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred)))
                scores['directional_accuracy'].append(dir_acc)
        
        return {metric: {'mean': np.mean(values), 'std': np.std(values)} 
                for metric, values in scores.items()}
    
    def _clone_model(self, model):
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            return type(model)(**model.get_params())
    
    def walk_forward_validation(self, model, X, y, initial_train_size=0.7, step_size=0.05):
        initial_size = int(len(X) * initial_train_size)
        step = int(len(X) * step_size)
        
        predictions = []
        actuals = []
        
        for start in range(initial_size, len(X) - step, step):
            end = start + step
            
            X_train = X[:start]
            y_train = y.iloc[:start]
            X_test = X[start:end]
            y_test = y.iloc[start:end]
            
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            
            y_pred = model_copy.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'mse': mean_squared_error(actuals, predictions),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'predictions': predictions,
            'actuals': actuals
        }

class AutoMLTrainer:
    def __init__(self, task_type='regression', time_budget=3600):
        self.task_type = task_type
        self.time_budget = time_budget
        self.best_pipeline = None
        
    def auto_train(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        start_time = datetime.now()
        
        model_configs = self._get_auto_model_configs()
        preprocessing_configs = self._get_preprocessing_configs()
        
        best_score = float('inf')
        best_config = None
        
        for preprocess_name, preprocessor in preprocessing_configs.items():
            for model_name, model_config in model_configs.items():
                if (datetime.now() - start_time).seconds > self.time_budget:
                    break
                
                try:
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model_config['model'])
                    ])
                    
                    if model_config['params']:
                        param_grid = {f'model__{k}': v for k, v in model_config['params'].items()}
                        
                        tscv = TimeSeriesSplit(n_splits=3)
                        grid_search = GridSearchCV(
                            pipeline, param_grid, cv=tscv, 
                            scoring='neg_mean_squared_error', n_jobs=1
                        )
                        grid_search.fit(X_train, y_train)
                        
                        score = -grid_search.best_score_
                        trained_pipeline = grid_search.best_estimator_
                    else:
                        pipeline.fit(X_train, y_train)
                        
                        if X_val is not None and y_val is not None:
                            y_pred = pipeline.predict(X_val)
                            score = mean_squared_error(y_val, y_pred)
                        else:
                            tscv = TimeSeriesSplit(n_splits=3)
                            scores = cross_val_score(pipeline, X_train, y_train, 
                                                   cv=tscv, scoring='neg_mean_squared_error')
                            score = -scores.mean()
                        
                        trained_pipeline = pipeline
                    
                    if score < best_score:
                        best_score = score
                        best_config = (preprocess_name, model_name)
                        self.best_pipeline = trained_pipeline
                    
                    print(f"{preprocess_name} + {model_name}: {score:.4f}")
                    
                except Exception as e:
                    print(f"Error with {preprocess_name} + {model_name}: {e}")
                    continue
        
        return {
            'best_pipeline': self.best_pipeline,
            'best_score': best_score,
            'best_config': best_config
        }
    
    def _get_auto_model_configs(self):
        return {
            'linear': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_estimators=50),
                'params': {
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 10]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42, n_estimators=50),
                'params': {
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 6]
                }
            }
        }
    
    def _get_preprocessing_configs(self):
        return {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
    
    def predict(self, X):
        if self.best_pipeline is None:
            raise ValueError("No pipeline trained. Call auto_train() first.")
        
        return self.best_pipeline.predict(X)

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.trainers = {}
        self.results = {}
        
    def setup_trainers(self):
        if 'traditional_ml' in self.config:
            self.trainers['traditional'] = ModelTrainer(
                task_type=self.config.get('task_type', 'regression')
            )
        
        if 'neural_network' in self.config:
            nn_config = self.config['neural_network']
            self.trainers['neural_network'] = NeuralNetworkTrainer(
                input_size=nn_config['input_size'],
                hidden_sizes=nn_config.get('hidden_sizes', [128, 64]),
                output_size=nn_config.get('output_size', 1),
                dropout=nn_config.get('dropout', 0.2)
            )
        
        if 'automl' in self.config:
            self.trainers['automl'] = AutoMLTrainer(
                task_type=self.config.get('task_type', 'regression'),
                time_budget=self.config['automl'].get('time_budget', 1800)
            )
    
    def run_training(self, X_train, y_train, X_val=None, y_val=None):
        self.setup_trainers()
        
        for trainer_name, trainer in self.trainers.items():
            print(f"\n=== Training {trainer_name} ===")
            
            try:
                if trainer_name == 'traditional':
                    model_names = self.config.get('traditional_ml', {}).get('models')
                    optimize_hyperparams = self.config.get('traditional_ml', {}).get('optimize', True)
                    
                    result = trainer.train_all_models(
                        X_train, y_train, X_val, y_val,
                        model_names=model_names,
                        optimize_hyperparams=optimize_hyperparams
                    )
                    
                elif trainer_name == 'neural_network':
                    nn_config = self.config['neural_network']
                    result = trainer.train(
                        X_train, y_train, X_val, y_val,
                        epochs=nn_config.get('epochs', 100),
                        batch_size=nn_config.get('batch_size', 32),
                        learning_rate=nn_config.get('learning_rate', 0.001),
                        patience=nn_config.get('patience', 10)
                    )
                    
                elif trainer_name == 'automl':
                    result = trainer.auto_train(X_train, y_train, X_val, y_val)
                
                self.results[trainer_name] = result
                print(f"{trainer_name} training completed")
                
            except Exception as e:
                print(f"Error training {trainer_name}: {e}")
                self.results[trainer_name] = {'error': str(e)}
        
        return self.results
    
    def evaluate_all(self, X_test, y_test):
        evaluation_results = {}
        
        for trainer_name, trainer in self.trainers.items():
            try:
                if trainer_name == 'traditional':
                    results = trainer.evaluate_all_models(X_test, y_test)
                    evaluation_results[trainer_name] = results
                    
                elif trainer_name == 'neural_network':
                    y_pred = trainer.predict(X_test)
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    evaluation_results[trainer_name] = metrics
                    
                elif trainer_name == 'automl':
                    y_pred = trainer.predict(X_test)
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    evaluation_results[trainer_name] = metrics
                
            except Exception as e:
                evaluation_results[trainer_name] = {'error': str(e)}
        
        return evaluation_results
    
    def get_best_model(self):
        best_model = None
        best_score = float('inf')
        best_trainer = None
        
        for trainer_name, trainer in self.trainers.items():
            if trainer_name == 'traditional' and hasattr(trainer, 'best_score'):
                if trainer.best_score < best_score:
                    best_score = trainer.best_score
                    best_model = trainer.best_model
                    best_trainer = trainer_name
            
            elif trainer_name in ['neural_network', 'automl']:
                if trainer_name in self.results:
                    result = self.results[trainer_name]
                    if 'best_val_loss' in result and result['best_val_loss'] < best_score:
                        best_score = result['best_val_loss']
                        best_model = trainer
                        best_trainer = trainer_name
                    elif 'best_score' in result and result['best_score'] < best_score:
                        best_score = result['best_score']
                        best_model = trainer
                        best_trainer = trainer_name
        
        return {
            'model': best_model,
            'trainer': best_trainer,
            'score': best_score
        }
    
    def save_all_models(self, base_path):
        for trainer_name, trainer in self.trainers.items():
            filepath = f"{base_path}_{trainer_name}.pkl"
            
            try:
                if trainer_name == 'traditional':
                    trainer.save_models(filepath)
                elif trainer_name == 'neural_network':
                    trainer.save_model(filepath)
                elif trainer_name == 'automl':
                    joblib.dump(trainer.best_pipeline, filepath)
                
                print(f"Saved {trainer_name} model to {filepath}")
                
            except Exception as e:
                print(f"Error saving {trainer_name}: {e}")