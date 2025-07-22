"""
🚀 Test Rapide des Algorithmes ML - Boubacar DABO
===============================================

Script pour tester rapidement tous les nouveaux algorithmes ML
et générer un rapport de performance.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Algorithmes de base
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Algorithmes avancés
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

def get_all_algorithms():
    """Retourner tous les algorithmes disponibles"""
    
    algorithms = {
        # Tree-based
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        
        # Boosting classique
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        
        # Linear
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000),
        
        # Instance-based
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        
        # Probabilistic
        'Gaussian Naive Bayes': GaussianNB(),
        
        # SVM
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        
        # Neural Networks
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
    }
    
    # Ajouter les algorithmes avancés
    if ADVANCED_AVAILABLE:
        algorithms.update({
            'XGBoost': xgb.XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False)
        })
    
    return algorithms

def test_algorithms():
    """Tester tous les algorithmes sur un dataset de test"""
    
    print("🚀 TEST RAPIDE DES ALGORITHMES ML")
    print("=" * 50)
    
    # Génération du dataset de test
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_classes=3, 
        random_state=42
    )
    
    # Préparation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset: {X.shape[0]} échantillons, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"🔄 Train: {len(X_train)}, Test: {len(X_test)}")
    print()
    
    # Tester tous les algorithmes
    results = []
    algorithms = get_all_algorithms()
    
    print(f"🤖 Test de {len(algorithms)} algorithmes...")
    print("-" * 50)
    
    for name, algorithm in algorithms.items():
        try:
            # Mesurer le temps d'entraînement
            start_time = time.time()
            
            # Cross-validation
            cv_scores = cross_val_score(algorithm, X_train, y_train, cv=5, scoring='accuracy')
            
            # Test final
            algorithm.fit(X_train, y_train)
            test_score = algorithm.score(X_test, y_test)
            
            train_time = time.time() - start_time
            
            # Stocker les résultats
            results.append({
                'Algorithm': name,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'Test_Accuracy': test_score,
                'Train_Time': train_time
            })
            
            print(f"✅ {name:20s} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | Test: {test_score:.3f} | Time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"❌ {name:20s} | Erreur: {str(e)}")
    
    return pd.DataFrame(results)

def generate_performance_report(results_df):
    """Générer un rapport de performance"""
    
    print("\n" + "=" * 60)
    print("📋 RAPPORT DE PERFORMANCE")
    print("=" * 60)
    
    # Statistiques générales
    print(f"🎯 Algorithmes testés: {len(results_df)}")
    print(f"📈 Accuracy moyenne: {results_df['Test_Accuracy'].mean():.4f}")
    print(f"📊 Écart-type: {results_df['Test_Accuracy'].std():.4f}")
    print(f"⏱️ Temps moyen: {results_df['Train_Time'].mean():.2f}s")
    print()
    
    # Top 5 des performants
    print("🏆 TOP 5 PERFORMANCE:")
    top_performance = results_df.nlargest(5, 'Test_Accuracy')
    for i, (_, row) in enumerate(top_performance.iterrows(), 1):
        print(f"  {i}. {row['Algorithm']:20s} | {row['Test_Accuracy']:.4f}")
    print()
    
    # Top 5 des rapides
    print("⚡ TOP 5 VITESSE:")
    top_speed = results_df.nsmallest(5, 'Train_Time')
    for i, (_, row) in enumerate(top_speed.iterrows(), 1):
        print(f"  {i}. {row['Algorithm']:20s} | {row['Train_Time']:.2f}s")
    print()
    
    # Algorithmes recommandés
    print("💡 RECOMMANDATIONS:")
    
    # Meilleur équilibre performance/vitesse
    results_df['Performance_Speed_Ratio'] = results_df['Test_Accuracy'] / results_df['Train_Time']
    best_ratio = results_df.loc[results_df['Performance_Speed_Ratio'].idxmax()]
    
    print(f"  🎯 Meilleur équilibre: {best_ratio['Algorithm']} (Score: {best_ratio['Test_Accuracy']:.3f}, Temps: {best_ratio['Train_Time']:.2f}s)")
    print(f"  🚀 Plus performant: {results_df.loc[results_df['Test_Accuracy'].idxmax(), 'Algorithm']} ({results_df['Test_Accuracy'].max():.4f})")
    print(f"  ⚡ Plus rapide: {results_df.loc[results_df['Train_Time'].idxmin(), 'Algorithm']} ({results_df['Train_Time'].min():.2f}s)")
    
    # Stabilité
    most_stable = results_df.loc[results_df['CV_Std'].idxmin()]
    print(f"  🎲 Plus stable: {most_stable['Algorithm']} (CV std: {most_stable['CV_Std']:.4f})")
    
    print()
    print("🔧 DISPONIBILITÉ DES LIBRAIRIES:")
    if ADVANCED_AVAILABLE:
        print("  ✅ XGBoost, LightGBM, CatBoost: Installés")
    else:
        print("  ⚠️ Librairies avancées manquantes - Installer avec:")
        print("     pip install xgboost lightgbm catboost")
    
    print()
    print("👨‍💼 Boubacar DABO | ESIGELEC - Big Data & IA")
    print("📧 dabom372@gmail.com | 🌐 https://bouba-dabo.github.io/portfolio")

def main():
    """Fonction principale"""
    
    # Tester les algorithmes
    results_df = test_algorithms()
    
    # Générer le rapport
    generate_performance_report(results_df)
    
    # Sauvegarder les résultats
    results_df.to_csv('ml_performance_test.csv', index=False)
    print(f"\n💾 Résultats sauvegardés dans: ml_performance_test.csv")
    
    return results_df

if __name__ == "__main__":
    results = main()
