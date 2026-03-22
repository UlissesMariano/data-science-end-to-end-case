import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

code_cells = []

# Cell 1: Intro
nb.cells.append(nbf.v4.new_markdown_cell("# Model Optimization\nEste notebook tem como foco realizar o treinamento de modelos de Machine Learning utilizando validação cruzada, otimização de hiperparâmetros (Hyperparameter Tuning com Scikit-Optimize) e seleção de variáveis (Feature Selection). Objetivo: prever `total_beneficios` com base em indicadores socioeconômicos."))

# Cell 2: Imports
imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel

# Para tuning Bayesiano
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import warnings
warnings.filterwarnings('ignore')"""
nb.cells.append(nbf.v4.new_code_cell(imports))

# Cell 3: Load Data
load_data = """# Carregar a base processada no nb05
df = pd.read_csv('data/base_final_v2.csv')

print(f"Shape: {df.shape}")
df.head()"""
nb.cells.append(nbf.v4.new_code_cell(load_data))

# Cell 4: Define Features and Target
define_vars = """# Definir a variável alvo
target = 'total_beneficios'

# Variáveis identificadoras ou de vazamento de dados (data leakage) que compõem o total
vars_remover = [
    'cod_municipio', 'municipio', target,
    'qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 
    'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp'
]

features = [col for col in df.columns if col not in vars_remover]

X = df[features].copy()
y = df[target].copy()

print(f"Features ({len(features)}): {features}")"""
nb.cells.append(nbf.v4.new_code_cell(define_vars))

# Cell 5: Tuning Functions
tuning_func = """#=========================================================
# FUNÇÕES DE OTIMIZAÇÃO DE HIPERPARÂMETROS (BAYESIANA)
#=========================================================
def tuning_random_forest(X_train, y_train, max_calls=15):
    print("Iniciando Tuning: RandomForestRegressor...")
    space = [
        Integer(10, 200, name='n_estimators'),
        Integer(2, 20, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf')
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        rmses = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmses.append(mean_squared_error(y_val, preds, squared=False))
            
        return np.mean(rmses) # O skopt minimiza essa função
    
    res = gp_minimize(objective, space, n_calls=max_calls, random_state=42)
    best_params = {
        'n_estimators': res.x[0],
        'max_depth': res.x[1],
        'min_samples_split': res.x[2],
        'min_samples_leaf': res.x[3]
    }
    print(f"Melhores parâmetros RF: {best_params} | RMSE (CV): {res.fun:.4f}")
    return RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

def tuning_lgbm(X_train, y_train, max_calls=15):
    print("Iniciando Tuning: LGBMRegressor...")
    space = [
        Real(0.01, 0.3, name='learning_rate'),
        Integer(20, 200, name='n_estimators'),
        Integer(2, 12, name='max_depth'),
        Integer(2, 30, name='min_child_samples') # em vez de num_leaves para lgbm
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        rmses = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmses.append(mean_squared_error(y_val, preds, squared=False))
        return np.mean(rmses)
    
    res = gp_minimize(objective, space, n_calls=max_calls, random_state=42)
    best_params = {
        'learning_rate': res.x[0],
        'n_estimators': res.x[1],
        'max_depth': res.x[2],
        'min_child_samples': res.x[3]
    }
    print(f"Melhores parâmetros LGBM: {best_params} | RMSE (CV): {res.fun:.4f}")
    return LGBMRegressor(**best_params, random_state=42, verbose=-1)

def tuning_xgboost(X_train, y_train, max_calls=15):
    print("Iniciando Tuning: XGBRegressor...")
    space = [
        Real(0.01, 0.3, name='learning_rate'),
        Integer(50, 300, name='n_estimators'),
        Integer(3, 10, name='max_depth'),
        Real(0.5, 1.0, name='subsample')
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = XGBRegressor(**params, random_state=42, n_jobs=-1, objective='reg:squarederror')
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        rmses = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_val)
            rmses.append(mean_squared_error(y_val, preds, squared=False))
        return np.mean(rmses)
    
    res = gp_minimize(objective, space, n_calls=max_calls, random_state=42)
    best_params = {
        'learning_rate': res.x[0],
        'n_estimators': res.x[1],
        'max_depth': res.x[2],
        'subsample': res.x[3]
    }
    print(f"Melhores parâmetros XGB: {best_params} | RMSE (CV): {res.fun:.4f}")
    return XGBRegressor(**best_params, random_state=42, n_jobs=-1, objective='reg:squarederror')"""
nb.cells.append(nbf.v4.new_code_cell(tuning_func))

# Cell 6: Feature Selection Function
feature_selection = """#=========================================================
# FUNÇÃO DE SELEÇÃO DE VARIÁVEIS (FEATURE SELECTION)
#=========================================================
def feature_selection(X_train, y_train, max_features=None):
    print("Executando Seleção de Features com RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    selector = SelectFromModel(rf, prefit=True, max_features=max_features, threshold=-np.inf if max_features else 'mean')
    features_selecionadas = X_train.columns[selector.get_support()]
    
    print(f"Variáveis originais: {X_train.shape[1]} | Selecionadas: {len(features_selecionadas)}")
    return list(features_selecionadas)"""
nb.cells.append(nbf.v4.new_code_cell(feature_selection))

# Cell 7: Main Training Function
treinar_modelo = """#=========================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO E AVALIAÇÃO
#=========================================================
def treinar_modelo(X, y, modelo_nome='rf', optimize_hyperparams=True, fs_max_features=None, n_splits=5):
    print(f"\\n{'='*50}\\nTREINANDO MODELO: {modelo_nome.upper()}\\n{'='*50}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rmses, maes, r2s = [], [], []
    feature_importances = []
    
    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"\\n--- Fold {fold}/{n_splits} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Feature Selection
        features_to_use = X.columns.tolist()
        if fs_max_features is not None:
            features_to_use = feature_selection(X_train, y_train, max_features=fs_max_features)
            X_train = X_train[features_to_use]
            X_test = X_test[features_to_use]
            
        # 2. Hyperparameter Tuning
        if optimize_hyperparams:
            if modelo_nome == 'rf':
                model = tuning_random_forest(X_train, y_train, max_calls=15)
            elif modelo_nome == 'lgbm':
                model = tuning_lgbm(X_train, y_train, max_calls=15)
            elif modelo_nome == 'xgb':
                model = tuning_xgboost(X_train, y_train, max_calls=15)
            else:
                raise ValueError("Modelo não suportado.")
        else:
            # Treinar com parâmetros default se não otimizado
            if modelo_nome == 'rf': model = RandomForestRegressor(random_state=42)
            elif modelo_nome == 'lgbm': model = LGBMRegressor(random_state=42)
            elif modelo_nome == 'xgb': model = XGBRegressor(random_state=42)
            
        # 3. Treinar Modelo Final do Fold
        model.fit(X_train, y_train)
        
        # 4. Avaliar Modelo
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        print(f"Resultado Fold {fold} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        
        # Salvar feature importances se suportado
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=features_to_use)
            feature_importances.append(importances)
            
        fold += 1
        
    print(f"\\n{'='*50}\\nRESUMO FINAL - {modelo_nome.upper()}\\n{'='*50}")
    print(f"Média RMSE: {np.mean(rmses):.2f} (+/- {np.std(rmses):.2f})")
    print(f"Média MAE : {np.mean(maes):.2f} (+/- {np.std(maes):.2f})")
    print(f"Média R²  : {np.mean(r2s):.4f} (+/- {np.std(r2s):.4f})")
    
    # Agregar importâncias
    if feature_importances:
        df_imp = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        print("\\nTop 10 Variáveis Mais Importantes:")
        print(df_imp.head(10))
        
    return {
        'rmse': np.mean(rmses),
        'mae': np.mean(maes),
        'r2': np.mean(r2s),
        'importances': df_imp if feature_importances else None
    }"""
nb.cells.append(nbf.v4.new_code_cell(treinar_modelo))

# Cell 8: Execution Random Forest
exec_rf = """# Testar com Random Forest (com Tuning reduzido e todas as features)
res_rf = treinar_modelo(X, y, modelo_nome='rf', optimize_hyperparams=True, fs_max_features=None, n_splits=3)"""
nb.cells.append(nbf.v4.new_code_cell(exec_rf))

# Cell 9: Execution LightGBM
exec_lgbm = """# Testar com LightGBM
res_lgbm = treinar_modelo(X, y, modelo_nome='lgbm', optimize_hyperparams=True, fs_max_features=None, n_splits=3)"""
nb.cells.append(nbf.v4.new_code_cell(exec_lgbm))

# Cell 10: Execution XGBoost
exec_xgb = """# Testar com XGBoost
res_xgb = treinar_modelo(X, y, modelo_nome='xgb', optimize_hyperparams=True, fs_max_features=None, n_splits=3)"""
nb.cells.append(nbf.v4.new_code_cell(exec_xgb))

# Save notebook
with open('nb06_model_optimization.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook nb06_model_optimization.ipynb gerado com sucesso!")
