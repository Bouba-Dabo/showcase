"""
🚀 Machine Learning Hub - Page d'accueil
=======================================

Interface principale pour naviguer entre les différentes applications ML.
"""

import streamlit as st
import subprocess
import os

# Configuration de la page
st.set_page_config(
    page_title="ML Hub - Boubacar DABO",
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
    
    .app-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .app-card:hover {
        transform: translateY(-5px);
    }
    
    .tech-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Machine Learning Showcase Hub</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 3rem;">
        <strong>Portfolio Technique - Boubacar DABO</strong><br>
        Démonstration complète de compétences en Data Science & Machine Learning
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques du portfolio
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-container">
            <h3>15+</h3>
            <p>Algorithmes ML</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-container">
            <h3>3</h3>
            <p>Apps Interactives</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-container">
            <h3>10+</h3>
            <p>Datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-container">
            <h3>5+</h3>
            <p>Visualisations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Applications disponibles
    st.markdown("## 🚀 Applications Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-card">
            <h3>📊 Dashboard Principal</h3>
            <p>Interface complète avec visualisations interactives, 
            analyse exploratoire et modélisation ML en temps réel.</p>
            
            <div style="margin: 1rem 0;">
                <span class="tech-badge">Random Forest</span>
                <span class="tech-badge">XGBoost</span>
                <span class="tech-badge">SVM</span>
                <span class="tech-badge">Neural Networks</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Lancer Dashboard", key="dashboard", use_container_width=True):
            st.switch_page("streamlit_apps/dashboard.py")
    
    with col2:
        st.markdown("""
        <div class="app-card">
            <h3>🧠 ML Avancé</h3>
            <p>Algorithmes de pointe avec optimisation d'hyperparamètres,
            validation croisée et comparaisons multi-modèles.</p>
            
            <div style="margin: 1rem 0;">
                <span class="tech-badge">LightGBM</span>
                <span class="tech-badge">CatBoost</span>
                <span class="tech-badge">Grid Search</span>
                <span class="tech-badge">Cross-Validation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧠 Lancer ML Avancé", key="advanced", use_container_width=True):
            st.switch_page("streamlit_apps/advanced_ml.py")
    
    # Section des algorithmes
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
        - **CatBoost** - Boosting pour variables catégorielles
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Linear & Boosting
        - **Logistic Regression** - Régression logistique
        - **SGD Classifier** - Descente de gradient stochastique
        - **Gradient Boosting** - Boosting de gradient
        - **AdaBoost** - Boosting adaptatif
        - **SVM** - Machine à vecteurs de support
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 Neural & Probabilistic
        - **Neural Network (MLP)** - Perceptron multicouche
        - **K-Nearest Neighbors** - K plus proches voisins
        - **Gaussian Naive Bayes** - Bayes naïf gaussien
        - **Multinomial Naive Bayes** - Bayes naïf multinomial
        """)
    
    # Technologies utilisées
    st.markdown("## 🛠️ Stack Technique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Libraries
        - **Python 3.13** - Langage principal
        - **Scikit-learn** - Framework ML principal
        - **Pandas & NumPy** - Manipulation de données
        - **Matplotlib & Seaborn** - Visualisations statiques
        - **Plotly** - Visualisations interactives
        """)
    
    with col2:
        st.markdown("""
        ### Advanced ML
        - **XGBoost** - Gradient boosting optimisé
        - **LightGBM** - ML ultra-rapide
        - **CatBoost** - Gestion automatique des catégories
        - **Streamlit** - Applications web interactives
        - **Joblib** - Parallélisation et persistence
        """)
    
    # Contact et liens
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
        
        🎯 Focus sur ML industriel et Data Engineering
        """)

if __name__ == "__main__":
    main()
