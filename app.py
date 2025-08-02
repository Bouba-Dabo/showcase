"""
🚀 Machine Learning Showcase Hub - Application Principale
========================================================

Point d'entrée principal pour l'ensemble des applications ML.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="ML Showcase - Boubacar DABO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .tech-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Machine Learning Showcase</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 3rem;">
        <strong>Portfolio Technique - Boubacar DABO</strong><br>
        Démonstration complète de compétences en Data Science & Machine Learning
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.info("""
    👈 **Utilisez la barre latérale pour naviguer** entre les différentes applications :
    - 📊 **Dashboard** : Interface principale avec visualisations
    - 🧠 **ML Avancé** : 15+ algorithmes avec configuration avancée
    - 🔧 **Optimisation Optuna** : Optimisation automatique des hyperparamètres
    """)
    
    # Statistiques du portfolio
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Algorithmes ML", "15+", "Implémentés")
    
    with col2:
        st.metric("📱 Apps Interactives", "4", "Fonctionnelles")
    
    with col3:
        st.metric("📊 Datasets", "10+", "Analysés")
    
    with col4:
        st.metric("📈 Visualisations", "5+", "Types")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fonctionnalités principales
    st.markdown("## 🚀 Fonctionnalités Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **📊 Dashboard Interactif**
        
        Interface complète avec visualisations en temps réel, analyse exploratoire automatisée et modélisation ML interactive.
        
        **Technologies:** Plotly • Scikit-learn • Pandas • Streamlit
        """)
    
    with col2:
        st.info("""
        **🧠 ML Avancé**
        
        12+ algorithmes de Machine Learning avec optimisation d'hyperparamètres, validation croisée et comparaisons multi-modèles.
        
        **Technologies:** Random Forest • SVM • Neural Networks • Gradient Boosting
        """)
    
    with col3:
        st.warning("""
        **🔧 Optimisation Optuna**
        
        Optimisation automatique des hyperparamètres avec algorithmes bayésiens et visualisations en temps réel.
        
        **Technologies:** Optuna • TPE Sampler • Hyperparamètre Tuning
        """)
    
    # Algorithmes implémentés
    st.markdown("## 🤖 Algorithmes Implémentés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🌳 Tree-Based
        - **Random Forest** - Ensemble d'arbres
        - **Extra Trees** - Arbres extrêmement randomisés  
        - **Decision Tree** - Arbre de décision simple
        - **XGBoost** - Gradient boosting optimisé
        - **LightGBM** - Gradient boosting rapide
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Linear & Boosting
        - **Logistic Regression** - Régression logistique
        - **SGD Classifier** - Descente de gradient
        - **Gradient Boosting** - Boosting de gradient
        - **AdaBoost** - Boosting adaptatif
        - **SVM** - Machine à vecteurs de support
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 Neural & Others
        - **Neural Network (MLP)** - Perceptron multicouche
        - **K-Nearest Neighbors** - K plus proches voisins
        - **Naive Bayes** - Algorithmes probabilistes
        - **CatBoost** - Boosting pour catégories
        """)
    
    # Informations de contact
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📧 Contact
        **Boubacar DABO**
        
        📧 dabom372@gmail.com
        
        ✅ **Disponible temps plein jusqu'en Septembre 2025**
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 Liens Professionnels
        
        💼 [LinkedIn](https://www.linkedin.com/in/boubacar-dabo-94206a291/)
        
        🌐 [Portfolio Web](https://bouba-dabo.github.io/portfolio)
        
        📂 [Code Source GitHub](https://github.com/Bouba-dabo/showcase)
        """)
    
    with col3:
        st.markdown("""
        ### 🎓 Formation
        
        **ESIGELEC** - École d'Ingénieurs
        
        Spécialisation **Big Data & Intelligence Artificielle**
        
        🎯 Focus sur ML industriel
        """)

if __name__ == "__main__":
    main()
