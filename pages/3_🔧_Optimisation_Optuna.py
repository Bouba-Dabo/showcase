"""
🔧 Optimisation Automatique des Hyperparamètres - Optuna
=======================================================

Interface professionnelle pour l'optimisation automatique des modèles ML
avec visualisations en temps réel et comparaisons de performances.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_iris, load_wine, load_breast_cancer
import time
from typing import Dict, Any, List, Tuple
import joblib
import os
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Optimisation Optuna - ML Showcase",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .optimization-status {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .best-params {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OptunaOptimizer:
    """Classe pour l'optimisation Optuna des modèles ML"""
    
    def __init__(self):
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def create_study(self, direction: str = "maximize", study_name: str = None):
        """Créer une nouvelle étude Optuna"""
        if study_name is None:
            study_name = f"ml_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
    def objective_random_forest(self, trial, X, y, cv_folds=5):
        """Fonction objectif pour Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return scores.mean()
    
    def objective_svm(self, trial, X, y, cv_folds=5):
        """Fonction objectif pour SVM"""
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly']) == 'rbf' else 'scale',
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'random_state': 42
        }
        
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
        
        model = SVC(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return scores.mean()
    
    def objective_logistic(self, trial, X, y, cv_folds=5):
        """Fonction objectif pour Logistic Regression"""
        params = {
            'C': trial.suggest_float('C', 0.01, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': 42
        }
        
        model = LogisticRegression(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return scores.mean()
    
    def optimize(self, objective_func, X, y, n_trials=100, timeout=None):
        """Lancer l'optimisation"""
        if self.study is None:
            self.create_study()
        
        # Créer la fonction objectif avec les données
        def objective(trial):
            return objective_func(trial, X, y)
        
        # Optimisation avec callback pour le suivi
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._optimization_callback]
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params, self.best_score
    
    def _optimization_callback(self, study, trial):
        """Callback pour suivre l'optimisation"""
        self.optimization_history.append({
            'trial': trial.number,
            'value': trial.value,
            'params': trial.params,
            'datetime': trial.datetime_start
        })
    
    def get_optimization_plots(self):
        """Générer les graphiques d'optimisation"""
        if not self.study:
            return None, None, None
        
        # 1. Historique d'optimisation
        fig_history = go.Figure()
        trials = [t.number for t in self.study.trials]
        values = [t.value for t in self.study.trials]
        
        fig_history.add_trace(go.Scatter(
            x=trials,
            y=values,
            mode='lines+markers',
            name='Score',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=6)
        ))
        
        fig_history.update_layout(
            title="Évolution du Score d'Optimisation",
            xaxis_title="Numéro du Trial",
            yaxis_title="Score de Validation Croisée",
            template="plotly_dark"
        )
        
        # 2. Importance des paramètres
        try:
            importance = optuna.importance.get_param_importances(self.study)
            fig_importance = go.Figure(go.Bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation='h',
                marker_color='#4ecdc4'
            ))
            
            fig_importance.update_layout(
                title="Importance des Hyperparamètres",
                xaxis_title="Importance",
                yaxis_title="Paramètre",
                template="plotly_dark"
            )
        except:
            fig_importance = None
        
        # 3. Distribution des paramètres
        if len(self.study.trials) > 10:
            param_names = list(self.best_params.keys())[:4]  # Top 4 paramètres
            fig_params = make_subplots(
                rows=2, cols=2,
                subplot_titles=param_names
            )
            
            for i, param in enumerate(param_names):
                row = i // 2 + 1
                col = i % 2 + 1
                
                values = [t.params.get(param) for t in self.study.trials if param in t.params]
                scores = [t.value for t in self.study.trials if param in t.params]
                
                fig_params.add_trace(
                    go.Scatter(x=values, y=scores, mode='markers', name=param),
                    row=row, col=col
                )
            
            fig_params.update_layout(
                title="Distribution des Paramètres vs Performance",
                template="plotly_dark",
                height=600
            )
        else:
            fig_params = None
            
        return fig_history, fig_importance, fig_params

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Charger un dataset"""
    if dataset_name == "Iris":
        data = load_iris()
        return data.data, data.target, data.feature_names
    elif dataset_name == "Wine":
        data = load_wine()
        return data.data, data.target, data.feature_names
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names
    elif dataset_name == "Synthétique":
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y, feature_names

def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 Optimisation Automatique Optuna</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
        <strong>Optimisation automatique des hyperparamètres</strong><br>
        Utilisez Optuna pour trouver automatiquement les meilleurs paramètres pour vos modèles ML
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'optimiseur dans le session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = OptunaOptimizer()
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection du dataset
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["Iris", "Wine", "Breast Cancer", "Synthétique"]
    )
    
    # Sélection de l'algorithme
    algorithm = st.sidebar.selectbox(
        "Algorithme ML",
        ["Random Forest", "SVM", "Logistic Regression"]
    )
    
    # Paramètres d'optimisation
    n_trials = st.sidebar.slider("Nombre de trials", 10, 200, 50)
    cv_folds = st.sidebar.slider("Folds de validation croisée", 3, 10, 5)
    timeout = st.sidebar.number_input("Timeout (secondes, 0 = illimité)", 0, 3600, 300)
    timeout = None if timeout == 0 else timeout
    
    # Chargement des données
    X, y, feature_names = load_dataset(dataset_name)
    
    # Informations sur le dataset
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Échantillons", X.shape[0])
    with col2:
        st.metric("🔢 Features", X.shape[1])
    with col3:
        st.metric("🎯 Classes", len(np.unique(y)))
    with col4:
        st.metric("⚖️ Balance", f"{(np.bincount(y).min() / len(y)):.2%}")
    
    # Bouton d'optimisation
    if st.sidebar.button("🚀 Lancer l'Optimisation", type="primary"):
        # Reset de l'optimiseur
        st.session_state.optimizer = OptunaOptimizer()
        
        # Sélection de la fonction objectif
        if algorithm == "Random Forest":
            objective_func = st.session_state.optimizer.objective_random_forest
        elif algorithm == "SVM":
            objective_func = st.session_state.optimizer.objective_svm
        else:
            objective_func = st.session_state.optimizer.objective_logistic
        
        # Conteneur pour le statut
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        with status_container.container():
            st.markdown('<div class="optimization-status">🔄 Optimisation en cours...</div>', unsafe_allow_html=True)
        
        # Lancement de l'optimisation
        start_time = time.time()
        
        try:
            best_params, best_score = st.session_state.optimizer.optimize(
                objective_func, X, y, n_trials=n_trials, timeout=timeout
            )
            
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Mise à jour du statut
            with status_container.container():
                st.markdown('<div class="optimization-status">✅ Optimisation terminée!</div>', unsafe_allow_html=True)
            
            progress_bar.progress(1.0)
            
            # Sauvegarde des résultats
            st.session_state.optimization_results = {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_time': optimization_time,
                'algorithm': algorithm,
                'dataset': dataset_name,
                'n_trials': len(st.session_state.optimizer.study.trials)
            }
            
        except Exception as e:
            st.error(f"Erreur lors de l'optimisation : {str(e)}")
    
    # Affichage des résultats
    if hasattr(st.session_state, 'optimization_results'):
        results = st.session_state.optimization_results
        
        st.markdown("## 🎯 Résultats de l'Optimisation")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Meilleur Score",
                f"{results['best_score']:.4f}",
                delta=f"+{(results['best_score'] - 0.5):.4f}"
            )
        
        with col2:
            st.metric(
                "⏱️ Temps d'Optimisation",
                f"{results['optimization_time']:.1f}s"
            )
        
        with col3:
            st.metric(
                "🔄 Trials Exécutés",
                results['n_trials']
            )
        
        with col4:
            st.metric(
                "🚀 Algorithme",
                results['algorithm']
            )
        
        # Meilleurs paramètres
        st.markdown('<div class="best-params">', unsafe_allow_html=True)
        st.markdown("### 🏆 Meilleurs Hyperparamètres")
        
        params_df = pd.DataFrame([
            {"Paramètre": k, "Valeur": v}
            for k, v in results['best_params'].items()
        ])
        st.dataframe(params_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graphiques d'optimisation
        if st.session_state.optimizer.study:
            st.markdown("## 📊 Visualisations de l'Optimisation")
            
            fig_history, fig_importance, fig_params = st.session_state.optimizer.get_optimization_plots()
            
            # Historique d'optimisation
            if fig_history:
                st.plotly_chart(fig_history, use_container_width=True)
            
            # Importance des paramètres et distribution
            if fig_importance or fig_params:
                col1, col2 = st.columns(2)
                
                if fig_importance:
                    with col1:
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                if fig_params:
                    with col2:
                        st.plotly_chart(fig_params, use_container_width=True)
    
    # Section informative
    st.markdown("---")
    st.markdown("## 🧠 Comment ça marche ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Optuna en Action
        
        1. **Échantillonnage Intelligent** - TPE Sampler
        2. **Évaluation Objective** - Validation croisée
        3. **Pruning Automatique** - Arrêt précoce
        4. **Optimisation Bayésienne** - Prédiction des zones prometteuses
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Avantages Professionnels
        
        - **Gain de temps** - Automation complète
        - **Performances optimales** - Meilleurs que le tuning manuel
        - **Reproductibilité** - Résultats cohérents
        - **Scalabilité** - Parallélisation possible
        """)

if __name__ == "__main__":
    main()
