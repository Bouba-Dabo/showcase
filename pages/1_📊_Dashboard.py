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
                             ["Random Forest", "XGBoost", "Support Vector Machine", 
                              "Neural Network (MLP)", "Gradient Boosting", "AdaBoost",
                              "Logistic Regression", "K-Nearest Neighbors", "Decision Tree",
                              "Extra Trees", "LightGBM", "CatBoost"])
    
    # Configuration spécifique selon le modèle
    col1, col2 = st.columns(2)
    
    if model_type == "Random Forest":
        with col1:
            n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
            max_depth = st.slider("Profondeur max", 1, 20, 10)
        with col2:
            min_samples_split = st.slider("Min samples split", 2, 10, 2)
            min_samples_leaf = st.slider("Min samples leaf", 1, 5, 2)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            random_state=random_state
        )
    
    elif model_type == "XGBoost":
        with col1:
            n_estimators = st.slider("Nombre d'estimateurs", 50, 300, 100)
            max_depth = st.slider("Profondeur max", 3, 10, 6)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1)
            subsample = st.slider("Subsample", 0.5, 1.0, 0.8)
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state
            )
        except ImportError:
            st.error("XGBoost non installé. Installation en cours...")
            st.code("pip install xgboost")
            st.stop()
    
    elif model_type == "Support Vector Machine":
        with col1:
            C = st.slider("Paramètre C", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        with col2:
            gamma = st.selectbox("Gamma", ["scale", "auto"])
            degree = st.slider("Degré (si poly)", 2, 5, 3) if kernel == "poly" else 3
        from sklearn.svm import SVC
        model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=random_state)
    
    elif model_type == "Neural Network (MLP)":
        with col1:
            hidden_layers = st.selectbox("Couches cachées", ["(100,)", "(100, 50)", "(200, 100, 50)", "(300, 200, 100)"])
            activation = st.selectbox("Fonction d'activation", ["relu", "tanh", "logistic"])
        with col2:
            learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.001, format="%.3f")
            max_iter = st.slider("Max iterations", 100, 1000, 200)
        
        # Convertir string en tuple
        hidden_layer_sizes = eval(hidden_layers)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=random_state
        )
    
    elif model_type == "Gradient Boosting":
        with col1:
            n_estimators = st.slider("Nombre d'estimateurs", 50, 200, 100)
            learning_rate = st.slider("Learning rate", 0.01, 0.2, 0.1)
        with col2:
            max_depth = st.slider("Profondeur max", 3, 8, 3)
            subsample = st.slider("Subsample", 0.5, 1.0, 1.0)
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state
        )
    
    elif model_type == "AdaBoost":
        with col1:
            n_estimators = st.slider("Nombre d'estimateurs", 10, 100, 50)
            learning_rate = st.slider("Learning rate", 0.1, 2.0, 1.0)
        with col2:
            algorithm = st.selectbox("Algorithme", ["SAMME", "SAMME.R"])
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )
    
    elif model_type == "Logistic Regression":
        with col1:
            C = st.slider("Paramètre C", 0.1, 10.0, 1.0)
            solver = st.selectbox("Solver", ["liblinear", "lbfgs", "saga"])
        with col2:
            penalty = st.selectbox("Pénalité", ["l2", "l1", "none"]) if solver == "saga" else "l2"
            max_iter = st.slider("Max iterations", 100, 1000, 100)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            C=C, solver=solver, penalty=penalty, max_iter=max_iter, random_state=random_state
        )
    
    elif model_type == "K-Nearest Neighbors":
        with col1:
            n_neighbors = st.slider("Nombre de voisins", 3, 15, 5)
            weights = st.selectbox("Poids", ["uniform", "distance"])
        with col2:
            metric = st.selectbox("Métrique", ["euclidean", "manhattan", "minkowski"])
            p = st.slider("Paramètre p (Minkowski)", 1, 3, 2) if metric == "minkowski" else 2
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric, p=p
        )
    
    elif model_type == "Decision Tree":
        with col1:
            max_depth = st.slider("Profondeur max", 1, 20, 5)
            criterion = st.selectbox("Critère", ["gini", "entropy"])
        with col2:
            min_samples_split = st.slider("Min samples split", 2, 10, 2)
            min_samples_leaf = st.slider("Min samples leaf", 1, 5, 1)
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
    
    elif model_type == "Extra Trees":
        with col1:
            n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
            max_depth = st.slider("Profondeur max", 1, 20, 10)
        with col2:
            min_samples_split = st.slider("Min samples split", 2, 10, 2)
            max_features = st.selectbox("Max features", ["sqrt", "log2", "auto"])
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state
        )
    
    elif model_type == "LightGBM":
        with col1:
            n_estimators = st.slider("Nombre d'estimateurs", 50, 300, 100)
            max_depth = st.slider("Profondeur max", 3, 15, 6)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1)
            num_leaves = st.slider("Nombre de feuilles", 10, 300, 31)
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=random_state,
                verbose=-1
            )
        except ImportError:
            st.error("LightGBM non installé. Installation en cours...")
            st.code("pip install lightgbm")
            st.stop()
    
    elif model_type == "CatBoost":
        with col1:
            iterations = st.slider("Nombre d'itérations", 50, 500, 100)
            depth = st.slider("Profondeur", 3, 10, 6)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1)
            l2_leaf_reg = st.slider("L2 leaf reg", 1, 10, 3)
        try:
            import catboost as cb
            model = cb.CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_state=random_state,
                verbose=False
            )
        except ImportError:
            st.error("CatBoost non installé. Installation en cours...")
            st.code("pip install catboost")
            st.stop()
    
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
