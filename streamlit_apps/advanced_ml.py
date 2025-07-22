"""
🚀 Advanced Machine Learning Showcase - Boubacar DABO
====================================================

Application dédiée aux algorithmes de Machine Learning avancés et modernes
pour démonstration complète des compétences techniques.

Author: Boubacar DABO
Date: Juillet 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Advanced ML Showcase - Boubacar DABO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_advanced_models():
    """Charger tous les modèles ML disponibles"""
    models = {}
    
    # Modèles de base toujours disponibles
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.neural_network import MLPClassifier
    
    models.update({
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier,
        "Extra Trees": ExtraTreesClassifier,
        "Logistic Regression": LogisticRegression,
        "Support Vector Machine": SVC,
        "K-Nearest Neighbors": KNeighborsClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "Gaussian Naive Bayes": GaussianNB,
        "Neural Network (MLP)": MLPClassifier,
        "SGD Classifier": SGDClassifier,
    })
    
    # Modèles avancés (optionnels)
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier
        st.sidebar.success("✅ XGBoost disponible")
    except ImportError:
        st.sidebar.warning("⚠️ XGBoost non installé")
    
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier
        st.sidebar.success("✅ LightGBM disponible")
    except ImportError:
        st.sidebar.warning("⚠️ LightGBM non installé")
    
    try:
        import catboost as cb
        models["CatBoost"] = cb.CatBoostClassifier
        st.sidebar.success("✅ CatBoost disponible")
    except ImportError:
        st.sidebar.warning("⚠️ CatBoost non installé")
    
    return models

def create_model_config(model_name):
    """Configuration des hyperparamètres pour chaque modèle"""
    if model_name == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("🌳 Nombre d'arbres", 10, 500, 100, key="rf_trees")
            max_depth = st.slider("📏 Profondeur max", 1, 30, 10, key="rf_depth")
        with col2:
            min_samples_split = st.slider("🔪 Min samples split", 2, 20, 2, key="rf_split")
            max_features = st.selectbox("🎯 Max features", ["sqrt", "log2", None], key="rf_features")
        
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "random_state": 42
        }
    
    elif model_name == "XGBoost":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("🚀 Nombre d'estimateurs", 50, 1000, 100, key="xgb_trees")
            max_depth = st.slider("📏 Profondeur max", 3, 15, 6, key="xgb_depth")
            learning_rate = st.slider("📈 Learning rate", 0.01, 0.5, 0.1, key="xgb_lr")
        with col2:
            subsample = st.slider("🎲 Subsample", 0.5, 1.0, 0.8, key="xgb_subsample")
            colsample_bytree = st.slider("🌿 Colsample by tree", 0.5, 1.0, 0.8, key="xgb_colsample")
            reg_alpha = st.slider("⚖️ Reg Alpha (L1)", 0.0, 1.0, 0.0, key="xgb_alpha")
        
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "random_state": 42,
            "verbosity": 0
        }
    
    elif model_name == "LightGBM":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("⚡ Nombre d'estimateurs", 50, 1000, 100, key="lgb_trees")
            max_depth = st.slider("📏 Profondeur max", 3, 15, 6, key="lgb_depth")
            learning_rate = st.slider("📈 Learning rate", 0.01, 0.5, 0.1, key="lgb_lr")
        with col2:
            num_leaves = st.slider("🍃 Nombre de feuilles", 10, 300, 31, key="lgb_leaves")
            subsample = st.slider("🎲 Subsample", 0.5, 1.0, 0.8, key="lgb_subsample")
            reg_alpha = st.slider("⚖️ Reg Alpha (L1)", 0.0, 1.0, 0.0, key="lgb_alpha")
        
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "reg_alpha": reg_alpha,
            "random_state": 42,
            "verbose": -1
        }
    
    elif model_name == "CatBoost":
        col1, col2 = st.columns(2)
        with col1:
            iterations = st.slider("🔄 Nombre d'itérations", 50, 1000, 100, key="cat_iter")
            depth = st.slider("📏 Profondeur", 3, 16, 6, key="cat_depth")
            learning_rate = st.slider("📈 Learning rate", 0.01, 0.5, 0.1, key="cat_lr")
        with col2:
            l2_leaf_reg = st.slider("🍃 L2 leaf reg", 1, 10, 3, key="cat_l2")
            border_count = st.slider("📊 Border count", 32, 255, 128, key="cat_border")
        
        return {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "border_count": border_count,
            "random_state": 42,
            "verbose": False
        }
    
    elif model_name == "Support Vector Machine":
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("💪 Paramètre C", 0.01, 100.0, 1.0, key="svm_c")
            kernel = st.selectbox("🔍 Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
        with col2:
            gamma = st.selectbox("📡 Gamma", ["scale", "auto"], key="svm_gamma")
            degree = st.slider("📐 Degré (poly)", 2, 5, 3, key="svm_degree") if kernel == "poly" else 3
        
        return {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
            "degree": degree,
            "random_state": 42
        }
    
    elif model_name == "Neural Network (MLP)":
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.selectbox("🧠 Architecture", 
                                       ["(100,)", "(100, 50)", "(200, 100, 50)", "(300, 200, 100)", "(500, 300, 100)"], 
                                       key="mlp_layers")
            activation = st.selectbox("⚡ Activation", ["relu", "tanh", "logistic"], key="mlp_activation")
        with col2:
            learning_rate = st.slider("📈 Learning rate", 0.0001, 0.1, 0.001, format="%.4f", key="mlp_lr")
            max_iter = st.slider("🔄 Max iterations", 100, 2000, 200, key="mlp_iter")
            alpha = st.slider("⚖️ Régularisation (alpha)", 0.0001, 0.01, 0.0001, format="%.4f", key="mlp_alpha")
        
        return {
            "hidden_layer_sizes": eval(hidden_layers),
            "activation": activation,
            "learning_rate_init": learning_rate,
            "max_iter": max_iter,
            "alpha": alpha,
            "random_state": 42
        }
    
    # Configuration simple pour les autres modèles
    else:
        return {"random_state": 42}

def display_model_comparison(results):
    """Afficher la comparaison des modèles"""
    
    # Créer un DataFrame des résultats
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('Accuracy', ascending=False)
    
    # Graphique de comparaison
    fig = px.bar(
        df_results.reset_index(), 
        x='index', 
        y='Accuracy',
        title="🏆 Comparaison des Performance des Modèles",
        color='Accuracy',
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_title="Modèles", yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.markdown("### 📊 Résultats détaillés")
    st.dataframe(df_results.round(4))

def main():
    """Application principale"""
    
    st.markdown("""
    # 🧠 Advanced Machine Learning Showcase
    
    ### Démonstration complète d'algorithmes ML modernes
    
    Cette application présente une gamme étendue d'algorithmes de Machine Learning,
    des modèles classiques aux techniques les plus récentes utilisées en industrie.
    """)
    
    # Sidebar pour les paramètres globaux
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Choix du dataset
    dataset_choice = st.sidebar.selectbox(
        "📊 Choisir le dataset",
        ["Synthétique", "Wine Quality", "Digits Recognition"]
    )
    
    # Génération des données
    if dataset_choice == "Synthétique":
        n_samples = st.sidebar.slider("Nombre d'échantillons", 100, 2000, 500)
        n_features = st.sidebar.slider("Nombre de features", 5, 20, 10)
        n_classes = st.sidebar.slider("Nombre de classes", 2, 5, 3)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            random_state=42
        )
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
    elif dataset_choice == "Wine Quality":
        wine = load_wine()
        X, y = wine.data, wine.target
        feature_names = wine.feature_names
        
    else:  # Digits
        digits = load_digits()
        X, y = digits.data, digits.target
        feature_names = [f"Pixel_{i+1}" for i in range(X.shape[1])]
    
    # Division train/test
    test_size = st.sidebar.slider("Taille du test set (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Standardisation
    if st.sidebar.checkbox("🔧 Standardiser les données", value=True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Informations sur le dataset
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Échantillons totaux", len(X))
    with col2:
        st.metric("🎯 Features", X.shape[1])
    with col3:
        st.metric("🏷️ Classes", len(np.unique(y)))
    with col4:
        st.metric("📈 Train/Test", f"{len(X_train)}/{len(X_test)}")
    
    # Charger les modèles disponibles
    available_models = load_advanced_models()
    
    # Interface principale
    tab1, tab2, tab3 = st.tabs(["🎯 Modèle Unique", "🏆 Comparaison Multiple", "📈 Optimisation"])
    
    with tab1:
        st.markdown("## 🎯 Entraînement d'un Modèle")
        
        model_name = st.selectbox("Choisir un algorithme", list(available_models.keys()))
        
        # Configuration du modèle
        st.markdown("### ⚙️ Configuration des Hyperparamètres")
        model_params = create_model_config(model_name)
        
        if st.button("🚀 Entraîner le Modèle"):
            with st.spinner("Entraînement en cours..."):
                # Créer et entraîner le modèle
                model_class = available_models[model_name]
                model = model_class(**model_params)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.4f}")
                    
                    # Matrice de confusion
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, title="Matrice de Confusion")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Rapport de classification
                    report = classification_report(y_test, y_pred, output_dict=True)
                    df_report = pd.DataFrame(report).transpose()
                    st.dataframe(df_report.round(3))
    
    with tab2:
        st.markdown("## 🏆 Comparaison Multiple")
        
        selected_models = st.multiselect(
            "Sélectionner les modèles à comparer",
            list(available_models.keys()),
            default=list(available_models.keys())[:5]
        )
        
        if st.button("🚀 Comparer les Modèles") and selected_models:
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Entraînement: {model_name}")
                
                try:
                    model_class = available_models[model_name]
                    model = model_class(random_state=42)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    
                    # Test final
                    model.fit(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    results[model_name] = {
                        "CV Mean": cv_scores.mean(),
                        "CV Std": cv_scores.std(),
                        "Accuracy": test_score
                    }
                    
                except Exception as e:
                    st.warning(f"Erreur avec {model_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("✅ Comparaison terminée!")
            display_model_comparison(results)
    
    with tab3:
        st.markdown("## 📈 Optimisation d'Hyperparamètres")
        
        model_for_tuning = st.selectbox(
            "Modèle à optimiser",
            ["Random Forest", "XGBoost", "SVM"],
            key="tuning_model"
        )
        
        if st.button("🔍 Lancer l'Optimisation"):
            with st.spinner("Optimisation en cours..."):
                if model_for_tuning == "Random Forest":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5, 10]
                    }
                    model = RandomForestClassifier(random_state=42)
                
                elif model_for_tuning == "XGBoost" and "XGBoost" in available_models:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                    import xgboost as xgb
                    model = xgb.XGBClassifier(random_state=42, verbosity=0)
                
                else:  # SVM
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                    model = SVC(random_state=42)
                
                # Grid Search
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Résultats
                st.success("✅ Optimisation terminée!")
                st.write("🏆 Meilleurs paramètres:", grid_search.best_params_)
                st.metric("🎯 Meilleur score CV", f"{grid_search.best_score_:.4f}")
                
                # Test avec les meilleurs paramètres
                best_score = grid_search.score(X_test, y_test)
                st.metric("📈 Score sur test set", f"{best_score:.4f}")

if __name__ == "__main__":
    main()
