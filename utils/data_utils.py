"""
Data Science Utilities - Boubacar DABO
======================================

Collection de fonctions utilitaires pour l'analyse de données,
le machine learning et la visualisation.

Author: Boubacar DABO
Date: Juillet 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Classe utilitaire pour l'analyse exploratoire de données.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'analyseur avec un DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame à analyser
        """
        self.df = df.copy()
        logger.info(f"DataAnalyzer initialisé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    def basic_info(self) -> Dict[str, Any]:
        """
        Retourne les informations de base du DataFrame.
        
        Returns:
            Dict contenant les statistiques de base
        """
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicates': self.df.duplicated().sum()
        }
        
        logger.info("Informations de base calculées")
        return info
    
    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calcule la matrice de corrélation pour les variables numériques.
        
        Args:
            method (str): Méthode de corrélation ('pearson', 'spearman', 'kendall')
        
        Returns:
            pd.DataFrame: Matrice de corrélation
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Moins de 2 colonnes numériques trouvées")
            return pd.DataFrame()
        
        corr_matrix = self.df[numeric_cols].corr(method=method)
        logger.info(f"Matrice de corrélation calculée avec la méthode {method}")
        
        return corr_matrix
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> List[int]:
        """
        Détecte les outliers dans une colonne spécifique.
        
        Args:
            column (str): Nom de la colonne
            method (str): Méthode de détection ('iqr', 'zscore')
        
        Returns:
            List[int]: Index des outliers
        """
        if column not in self.df.columns:
            raise ValueError(f"Colonne '{column}' non trouvée")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Colonne '{column}' n'est pas numérique")
        
        data = self.df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3].index.tolist()
        
        else:
            raise ValueError("Méthode doit être 'iqr' ou 'zscore'")
        
        logger.info(f"{len(outliers)} outliers détectés dans '{column}' avec la méthode {method}")
        return outliers

class MLEvaluator:
    """
    Classe utilitaire pour l'évaluation de modèles de machine learning.
    """
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, 
                               target_names: List[str] = None) -> Dict[str, Any]:
        """
        Évalue un modèle de classification.
        
        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_pred (np.ndarray): Prédictions
            target_names (List[str]): Noms des classes
        
        Returns:
            Dict contenant les métriques d'évaluation
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=target_names,
                                                         output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        logger.info("Évaluation de classification terminée")
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None, 
                             normalize: bool = False) -> go.Figure:
        """
        Crée un graphique de matrice de confusion interactif.
        
        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_pred (np.ndarray): Prédictions
            class_names (List[str]): Noms des classes
            normalize (bool): Normaliser la matrice
        
        Returns:
            go.Figure: Figure Plotly
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Matrice de confusion normalisée"
            text_template = '%{z:.2%}'
        else:
            title = "Matrice de confusion"
            text_template = '%{z:d}'
        
        if class_names is None:
            class_names = [f"Classe {i}" for i in range(len(cm))]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate=text_template,
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Prédictions",
            yaxis_title="Vraies étiquettes",
            height=500
        )
        
        return fig

class DataVisualizer:
    """
    Classe utilitaire pour la création de visualisations avancées.
    """
    
    @staticmethod
    def create_distribution_plot(data: pd.Series, title: str = None) -> go.Figure:
        """
        Crée un graphique de distribution avec histogramme et courbe de densité.
        
        Args:
            data (pd.Series): Données à visualiser
            title (str): Titre du graphique
        
        Returns:
            go.Figure: Figure Plotly
        """
        if title is None:
            title = f"Distribution de {data.name}"
        
        fig = go.Figure()
        
        # Histogramme
        fig.add_trace(go.Histogram(
            x=data,
            name="Histogramme",
            opacity=0.7,
            nbinsx=30
        ))
        
        # Courbe de densité
        from scipy import stats
        density = stats.gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density(x_range) * len(data) * (data.max() - data.min()) / 30,
            mode='lines',
            name="Densité",
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=data.name,
            yaxis_title="Fréquence",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
        """
        Crée une heatmap de corrélation interactive.
        
        Args:
            corr_matrix (pd.DataFrame): Matrice de corrélation
        
        Returns:
            go.Figure: Figure Plotly
        """
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Matrice de corrélation",
            height=500,
            width=500
        )
        
        return fig

class DataPreprocessor:
    """
    Classe utilitaire pour le préprocessing des données.
    """
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans un DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame avec valeurs manquantes
            strategy (str): Stratégie ('mean', 'median', 'mode', 'drop')
        
        Returns:
            pd.DataFrame: DataFrame traité
        """
        df_processed = df.copy()
        
        if strategy == 'drop':
            df_processed = df_processed.dropna()
        
        else:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            
            # Traitement des colonnes numériques
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    if strategy == 'mean':
                        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
            
            # Traitement des colonnes catégorielles
            for col in categorical_cols:
                if df_processed[col].isnull().any():
                    mode_value = df_processed[col].mode()
                    if len(mode_value) > 0:
                        df_processed[col].fillna(mode_value[0], inplace=True)
        
        logger.info(f"Valeurs manquantes traitées avec la stratégie '{strategy}'")
        return df_processed
    
    @staticmethod
    def scale_features(X_train: np.ndarray, X_test: np.ndarray = None, 
                      method: str = 'standard') -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Normalise les caractéristiques.
        
        Args:
            X_train (np.ndarray): Données d'entraînement
            X_test (np.ndarray): Données de test (optionnel)
            method (str): Méthode de normalisation ('standard', 'minmax')
        
        Returns:
            Tuple: (X_train_scaled, X_test_scaled, scaler)
        """
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("Méthode doit être 'standard' ou 'minmax'")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        
        logger.info(f"Normalisation effectuée avec la méthode '{method}'")
        return X_train_scaled, X_test_scaled, scaler

# Fonctions utilitaires globales
def load_and_validate_data(file_path: str, required_columns: List[str] = None) -> pd.DataFrame:
    """
    Charge et valide un fichier de données.
    
    Args:
        file_path (str): Chemin vers le fichier
        required_columns (List[str]): Colonnes requises
    
    Returns:
        pd.DataFrame: DataFrame validé
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non supporté")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        logger.info(f"Données chargées avec succès: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise

def generate_model_report(model, X_test: np.ndarray, y_test: np.ndarray, 
                         feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Génère un rapport complet pour un modèle ML.
    
    Args:
        model: Modèle entraîné
        X_test (np.ndarray): Données de test
        y_test (np.ndarray): Étiquettes de test
        feature_names (List[str]): Noms des caractéristiques
    
    Returns:
        Dict: Rapport complet du modèle
    """
    y_pred = model.predict(X_test)
    
    report = {
        'model_type': type(model).__name__,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred.tolist(),
        'evaluation': MLEvaluator.evaluate_classification(y_test, y_pred)
    }
    
    # Importance des caractéristiques si disponible
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        if feature_names:
            report['feature_importance'] = dict(zip(feature_names, importance))
        else:
            report['feature_importance'] = importance.tolist()
    
    logger.info("Rapport de modèle généré")
    return report

if __name__ == "__main__":
    # Tests basiques
    print("✅ Module data_utils importé avec succès")
    print("🔧 Classes disponibles: DataAnalyzer, MLEvaluator, DataVisualizer, DataPreprocessor")
    print("📊 Fonctions utilitaires prêtes pour l'analyse ML avancée")
