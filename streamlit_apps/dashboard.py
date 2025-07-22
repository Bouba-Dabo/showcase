"""
🚀 Data Science Dashboard - Boubacar DABO
========================================

Application Streamlit de démonstration des compétences en Data Science
et Machine Learning pour portfolio professionnel.

Author: Boubacar DABO
Date: Juillet 2025
Portfolio: https://bouba-dabo.github.io/portfolio
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Data Science Showcase - Boubacar DABO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design professionnel
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🚀 Data Science Showcase</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>Démonstration des compétences ML & IA - Boubacar DABO</h3>
        <p><em>Étudiant-ingénieur Big Data & Intelligence Artificielle @ ESIGELEC</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar pour navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choisir une démonstration",
        ["🏠 Accueil", "📊 Analyse Exploratoire", "🤖 Classification ML", "📈 Visualisations Avancées"]
    )

    if page == "🏠 Accueil":
        show_home()
    elif page == "📊 Analyse Exploratoire":
        show_data_analysis()
    elif page == "🤖 Classification ML":
        show_ml_classification()
    elif page == "📈 Visualisations Avancées":
        show_advanced_viz()

def show_home():
    """Page d'accueil avec présentation"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎯 À propos de cette démonstration</h2>', unsafe_allow_html=True)
        st.write("""
        Cette application interactive présente mes compétences en **Data Science** et **Machine Learning** 
        à travers des exemples concrets et professionnels.
        
        **Fonctionnalités incluses :**
        - 📊 Analyse exploratoire de données automatisée
        - 🤖 Classification avec plusieurs algorithmes ML
        - 📈 Visualisations interactives avec Plotly
        - ⚙️ Optimisation d'hyperparamètres en temps réel
        - 🔍 Métriques de performance détaillées
        """)
        
        st.markdown('<h2 class="sub-header">🛠️ Technologies utilisées</h2>', unsafe_allow_html=True)
        
        # Métriques techniques
        col_tech1, col_tech2, col_tech3 = st.columns(3)
        
        with col_tech1:
            st.metric("🐍 Python", "3.9+", "Langage principal")
        with col_tech2:
            st.metric("🤖 Scikit-learn", "1.1+", "Machine Learning")
        with col_tech3:
            st.metric("📊 Streamlit", "1.15+", "Interface web")

    with col2:
        st.markdown('<h2 class="sub-header">📧 Contact</h2>', unsafe_allow_html=True)
        st.info("""
        **Boubacar DABO**
        
        📧 dabom372@gmail.com
        
        💼 [LinkedIn](https://www.linkedin.com/in/boubacar-dabo-94206a291/)
        
        🌐 [Portfolio](https://bouba-dabo.github.io/portfolio)
        
        🎓 ESIGELEC - Big Data & IA
        """)
        
        # Statut de disponibilité
        st.success("✅ Disponible temps plein jusqu'en Septembre 2025")

def show_data_analysis():
    """Démonstration d'analyse exploratoire"""
    
    st.markdown('<h2 class="sub-header">📊 Analyse Exploratoire de Données</h2>', unsafe_allow_html=True)
    
    # Paramètres en dehors de la fonction mise en cache
    n_samples = st.slider("Nombre d'échantillons", 100, 1000, 500)
    
    # Génération de données synthétiques
    @st.cache_data
    def generate_sample_data(n_samples):
        # Utiliser le générateur numpy moderne
        rng = np.random.default_rng(42)
        
        data = {
            'age': rng.normal(35, 10, n_samples),
            'salary': rng.normal(50000, 15000, n_samples),
            'experience': rng.exponential(5, n_samples),
            'performance': rng.beta(2, 5, n_samples) * 100,
            'department': rng.choice(['IT', 'Marketing', 'Sales', 'HR'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    df = generate_sample_data(n_samples)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Aperçu des données")
        st.dataframe(df.head(10))
        
        st.subheader("📈 Statistiques descriptives")
        st.dataframe(df.describe())
    
    with col2:
        st.subheader("🎯 Distribution des variables")
        
        # Sélection de variable
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Choisir une variable", numeric_cols)
        
        # Histogramme interactif
        fig = px.histogram(df, x=selected_col, nbins=30, 
                          title=f"Distribution de {selected_col}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Matrice de corrélation
    st.subheader("🔗 Matrice de corrélation")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig = px.imshow(corr_matrix, 
                    labels={"color": "Corrélation"},
                    title="Matrice de corrélation des variables numériques")
    st.plotly_chart(fig, use_container_width=True)

def show_ml_classification():
    """Démonstration de classification ML"""
    
    st.markdown('<h2 class="sub-header">🤖 Classification Machine Learning</h2>', unsafe_allow_html=True)
    
    # Paramètres du dataset
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Nombre d'échantillons", 100, 1000, 300)
        n_features = st.slider("Nombre de caractéristiques", 2, 10, 4)
        n_classes = st.slider("Nombre de classes", 2, 5, 3)
    
    with col2:
        test_size = st.slider("Taille du test set (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random state", 0, 100, 42)
    
    # Génération des données
    @st.cache_data
    def generate_classification_data(n_samples, n_features, n_classes, random_state):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            random_state=random_state
        )
        return X, y
    
    X, y = generate_classification_data(n_samples, n_features, n_classes, random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Entraînement du modèle
    st.subheader("⚙️ Configuration du modèle")
    
    model_type = st.selectbox("Choisir un algorithme", 
                             ["Random Forest", "Logistic Regression", "SVM"])
    
    if model_type == "Random Forest":
        n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
        max_depth = st.slider("Profondeur max", 1, 20, 10)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state
        )
    
    # Entraînement et prédiction
    if st.button("🚀 Entraîner le modèle"):
        with st.spinner("Entraînement en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        # Affichage des résultats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("📊 Échantillons train", len(X_train))
        with col3:
            st.metric("🔍 Échantillons test", len(X_test))
        
        # Rapport de classification
        st.subheader("📋 Rapport détaillé")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Matrice de confusion
        if n_classes <= 4:  # Limiter l'affichage pour la lisibilité
            st.subheader("🔢 Matrice de confusion")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                           title="Matrice de confusion")
            st.plotly_chart(fig, use_container_width=True)

def show_advanced_viz():
    """Visualisations avancées"""
    
    st.markdown('<h2 class="sub-header">📈 Visualisations Avancées</h2>', unsafe_allow_html=True)
    
    # Dataset Iris pour les visualisations
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target_names[iris.target]
    
    tab1, tab2, tab3 = st.tabs(["🌸 Dataset Iris", "📊 Graphiques 3D", "🎯 Analyse de clusters"])
    
    with tab1:
        st.subheader("🌸 Analyse du dataset Iris")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot interactif
            fig = px.scatter(iris_df, 
                           x='sepal length (cm)', 
                           y='sepal width (cm)',
                           color='species',
                           size='petal length (cm)',
                           hover_data=['petal width (cm)'],
                           title="Relation Sepal Length vs Width")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            feature = st.selectbox("Choisir une caractéristique", iris_df.columns[:-1])
            fig = px.box(iris_df, x='species', y=feature,
                        title=f"Distribution de {feature} par espèce")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Visualisation 3D")
        
        # Graphique 3D
        fig = px.scatter_3d(iris_df,
                           x='sepal length (cm)',
                           y='sepal width (cm)', 
                           z='petal length (cm)',
                           color='species',
                           size='petal width (cm)',
                           title="Visualisation 3D du dataset Iris")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Heatmap de corrélation avancée")
        
        # Heatmap avec annotations
        numeric_data = iris_df.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.to_numpy(),
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            text=corr.round(3).to_numpy(),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Heatmap de corrélation - Dataset Iris",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
def show_footer():
    st.markdown("""
    <div class="footer">
        <h4>🚀 Data Science Showcase - Boubacar DABO</h4>
        <p>Portfolio technique démontrant l'expertise en Machine Learning et Intelligence Artificielle</p>
        <p><em>Développé avec Streamlit | Python | Plotly</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
