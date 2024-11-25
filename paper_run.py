import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# 加载已训练模型的数据
file_path = 'try.xlsx'  # 请将 'try.xlsx' 替换为你的训练数据文件路径
data = pd.read_excel(file_path)

# 提取特征和标签
X_text = data['名稱']
y = data['組別']

# 文本特征提取（TF-IDF）
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# 定义模型和参数网格
pipelines = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'SVM': SVC(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'Naive Bayes': MultinomialNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=300)
}

param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    },
    'Naive Bayes': {},
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh']
    }
}

# 通过 GridSearchCV 寻找最佳模型参数
best_models = {}
for model_name, model in pipelines.items():
    print(f"Optimizing {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_}")

# 加载新的待分类数据集
new_data_path = 'sample.xlsx'  # 请将 'sample.xlsx' 替换为你想要分类的新数据文件路径
new_data = pd.read_excel(new_data_path)

# 提取新数据的特征
X_new_text = new_data['名稱']
X_new = vectorizer.transform(X_new_text)

# 使用最优模型进行预测
for model_name, model in best_models.items():
    predictions = model.predict(X_new)
    new_data[f'{model_name} Prediction'] = predictions

# 输出或保存结果
output_path = 'classified_results.xlsx'  # 你可以更改输出文件路径
new_data.to_excel(output_path, index=False)

print("Predictions have been made and saved to", output_path)
