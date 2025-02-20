# Importar bibliotecas necessárias
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o conjunto de dados
print("1. Carregando o conjunto de dados...")
in_vehicle_coupon_recommendation = fetch_ucirepo(id=603)
X = in_vehicle_coupon_recommendation.data.features
y = in_vehicle_coupon_recommendation.data.targets

# 2. Análise Exploratória dos Dados (EDA)
print("2. Análise Exploratória dos Dados (EDA)...")
# Informações básicas sobre o conjunto de dados
print("Informações básicas sobre o conjunto de dados:")
print(X.info())

# Verificar valores faltantes
print("\nValores faltantes por coluna:")
print(X.isnull().sum())

# Distribuição das variáveis categóricas
print("\nDistribuição das variáveis categóricas:")
for column in X.select_dtypes(include=['object']).columns:
    print(f"\nDistribuição de {column}:")
    print(X[column].value_counts())

# Distribuição das variáveis numéricas
print("\nDistribuição das variáveis numéricas:")
X.select_dtypes(include=['int64', 'float64']).hist(bins=15, figsize=(15, 10), layout=(4, 4))
plt.suptitle('Distribuição das Variáveis Numéricas')
plt.tight_layout()
plt.show()

# Correlação entre variáveis numéricas
print("\nCorrelação entre variáveis numéricas:")
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = X[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# 3. Tratamento dos dados
print("3. Tratando valores faltantes e codificando variáveis categóricas...")
# Verificar e tratar valores faltantes
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Verificar valores faltantes após a imputação
print("\nValores faltantes após a imputação:")
print(X.isnull().sum())

# Codificar variáveis categóricas
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Codificar a variável alvo se necessário
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Assume que a primeira coluna é a variável alvo
if y.dtype == 'object':
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

# 4. Divisão dos dados
print("4. Dividindo os dados em conjuntos de treinamento e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Validação cruzada e otimização dos hiperparâmetros
print("5. Realizando validação cruzada e otimização dos hiperparâmetros...")

# Definir os hiperparâmetros a serem testados
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_et = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search para Floresta Aleatória
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
print(f'Melhores hiperparâmetros para Floresta Aleatória: {grid_search_rf.best_params_}')
print(f'Melhor pontuação de validação cruzada: {grid_search_rf.best_score_:.2f}')

# Grid Search para Árvores Extremamente Aleatórias
et_model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
grid_search_et = GridSearchCV(estimator=et_model, param_grid=param_grid_et, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_et.fit(X_train, y_train)
print(f'Melhores hiperparâmetros para Árvores Extremamente Aleatórias: {grid_search_et.best_params_}')
print(f'Melhor pontuação de validação cruzada: {grid_search_et.best_score_:.2f}')

# 6. Treinamento dos modelos com os melhores hiperparâmetros
print("6. Treinando os modelos com os melhores hiperparâmetros...")
best_rf_model = grid_search_rf.best_estimator_
best_et_model = grid_search_et.best_estimator_

best_rf_model.fit(X_train, y_train)
best_et_model.fit(X_train, y_train)

# 7. Avaliação dos resultados
print("7. Avaliando os modelos...")
# Previsões e precisão para Floresta Aleatória
rf_pred = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Precisão da Floresta Aleatória: {rf_accuracy:.2f}')

# Previsões e precisão para Árvores Extremamente Aleatórias
et_pred = best_et_model.predict(X_test)
et_accuracy = accuracy_score(y_test, et_pred)
print(f'Precisão das Árvores Extremamente Aleatórias: {et_accuracy:.2f}')

# 8. Importância dos atributos
print("8. Analisando a importância dos atributos...")
# Importância dos atributos usando Floresta Aleatória
rf_importances = best_rf_model.feature_importances_
for feature, importance in zip(X.columns, rf_importances):
    print(f'Atributo: {feature}, Importância na Floresta Aleatória: {importance:.2f}')

# Importância dos atributos usando Árvores Extremamente Aleatórias
et_importances = best_et_model.feature_importances_
for feature, importance in zip(X.columns, et_importances):
    print(f'Atributo: {feature}, Importância nas Árvores Extremamente Aleatórias: {importance:.2f}')

# 9. Exemplo de árvore totalmente aleatória
print("9. Exemplo de árvore totalmente aleatória...")
def totally_random_tree_split(X, y, depth=0, max_depth=10):
    n_samples, n_features = X.shape
    if n_samples <= 1 or depth >= max_depth:
        return np.mean(y)
    
    feature = np.random.randint(n_features)
    threshold = np.random.uniform(X[:, feature].min(), X[:, feature].max())
    left_mask = X[:, feature] < threshold
    right_mask = ~left_mask
    
    # Verificar se há amostras em ambos os lados
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return np.mean(y)
    
    left_tree = totally_random_tree_split(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_tree = totally_random_tree_split(X[right_mask], y[right_mask], depth + 1, max_depth)
    
    return (feature, threshold, left_tree, right_tree)

# Definir um limite máximo de profundidade para evitar recursão infinita
max_depth = 10
tree = totally_random_tree_split(X_train.values, y_train, max_depth=max_depth)
print(f'Exemplo de árvore totalmente aleatória: {tree}')