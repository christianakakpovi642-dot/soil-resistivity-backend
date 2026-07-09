"""
PREDICTION DE RESISTIVITE DU SOL - METHODE WENNER
MODELE HYBRIDE: RANDOM FOREST + KNN AVEC VALIDATION CROISEE
"""

import pandas as pd
import numpy as np
import os
import joblib
import folium
from folium import plugins
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
import time
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# CONSTANTES
# ================================
FICHIER_DONNEES = "resistivite_wenner.csv"
FICHIER_HISTORIQUE = "historique_predictions_wenner.csv"
FICHIER_HISTORIQUE_INITIAL = "historique_initial.csv"
FICHIER_MODELE = "modele_hybride_wenner.pkl"
FICHIER_SCALER = "scaler_wenner.pkl"
FICHIER_TRAIN_TEST = "train_test_split.pkl"
FICHIER_DERNIER_ENTRAINEMENT = "dernier_entrainement.txt"
FICHIER_CV_RESULTS = "cross_validation_results.pkl"
SEUIL_REENTRAINEMENT = 20  # Augmente pour eviter les reentraînements trop frequents
SEUIL_PERFORMANCE = 0.60   # Baisse pour etre plus realiste
INTERVALLE_ENTRAINEMENT_DEFAUT = 7
POIDS_RF = 0.7             # Poids optimise par la recherche
POIDS_KNN = 0.3
NOMBRE_PLIS_CV = 5
TAILLE_MAX_HISTORIQUE = 100  # Limite la taille de l'historique

# Variables globales pour stocker les ensembles de donnees
X_train_global = None
X_test_global = None
y_train_global = None
y_test_global = None

print("="*60)
print("PREDICTION DE RESISTIVITE DU SOL - METHODE WENNER")
print("MODELE HYBRIDE: RANDOM FOREST + KNN AVEC VALIDATION CROISEE")
print("="*60)

# ================================
# CLASSE DU MODELE HYBRIDE
# ================================

class HybridRegressor(BaseEstimator, RegressorMixin):
    """
    Modele hybride combinant Random Forest et KNN
    """
    def __init__(self, rf_weight=0.6, knn_weight=0.4, 
                 rf_n_estimators=200, rf_max_depth=20, rf_min_samples_split=5,
                 rf_min_samples_leaf=2, knn_n_neighbors=5, knn_weights='distance'):
        self.rf_weight = rf_weight
        self.knn_weight = knn_weight
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        
        self.rf_model = None
        self.knn_model = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """Entraîne les deux modeles"""
        # Entraînement du Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        
        # Entraînement du KNN
        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.knn_n_neighbors,
            weights=self.knn_weights,
            n_jobs=-1
        )
        self.knn_model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Prédiction par moyenne ponderee des deux modeles"""
        if not self.is_fitted:
            raise ValueError("Le modele n'a pas encore ete entraîne")
        
        rf_pred = self.rf_model.predict(X)
        knn_pred = self.knn_model.predict(X)
        
        # Combinaison ponderee
        predictions = self.rf_weight * rf_pred + self.knn_weight * knn_pred
        return predictions
    
    def get_params(self, deep=True):
        """Retourne les parametres du modele"""
        return {
            'rf_weight': self.rf_weight,
            'knn_weight': self.knn_weight,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'rf_min_samples_split': self.rf_min_samples_split,
            'rf_min_samples_leaf': self.rf_min_samples_leaf,
            'knn_n_neighbors': self.knn_n_neighbors,
            'knn_weights': self.knn_weights
        }
    
    def set_params(self, **params):
        """Definit les parametres du modele"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_feature_importances(self):
        """Retourne les importances des caracteristiques (moyenne ponderee)"""
        if not self.is_fitted:
            return None
        
        rf_importances = self.rf_model.feature_importances_
        knn_importances = np.ones_like(rf_importances) / len(rf_importances)
        
        # Moyenne ponderee
        hybrid_importances = self.rf_weight * rf_importances + self.knn_weight * knn_importances
        return hybrid_importances

# ================================
# FONCTIONS DE VALIDATION CROISEE
# ================================

def evaluer_validation_croisee(model, X, y, cv=NOMBRE_PLIS_CV, scoring='r2'):
    """
    Effectue une validation croisee du modele et retourne les resultats
    """
    print("\n" + "="*60)
    print(f"VALIDATION CROISEE ({cv} PLIS)")
    print("="*60)
    
    try:
        # Definir les metriques d'evaluation
        scorers = {
            'r2': make_scorer(r2_score),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), 
                               greater_is_better=False)
        }
        
        # Effectuer la validation croisee
        cv_results = cross_validate(
            model, X, y, 
            cv=cv, 
            scoring=scorers,
            n_jobs=-1,
            return_train_score=True,
            verbose=0
        )
        
        # Extraire les resultats
        train_r2_scores = cv_results['train_r2']
        test_r2_scores = cv_results['test_r2']
        test_mae_scores = -cv_results['test_mae']
        test_rmse_scores = -cv_results['test_rmse']
        
        # Afficher les resultats
        print("\nRESULTATS DE LA VALIDATION CROISEE:")
        print("-"*40)
        
        print("\nR2 SCORES:")
        print(f"   Train moyen: {np.mean(train_r2_scores):.4f} (+/- {np.std(train_r2_scores):.4f})")
        print(f"   Test moyen: {np.mean(test_r2_scores):.4f} (+/- {np.std(test_r2_scores):.4f})")
        print(f"   Min: {np.min(test_r2_scores):.4f}, Max: {np.max(test_r2_scores):.4f}")
        
        print("\nMAE SCORES (Ohm.m):")
        print(f"   Moyen: {np.mean(test_mae_scores):.2f} (+/- {np.std(test_mae_scores):.2f})")
        print(f"   Min: {np.min(test_mae_scores):.2f}, Max: {np.max(test_mae_scores):.2f}")
        
        print("\nRMSE SCORES (Ohm.m):")
        print(f"   Moyen: {np.mean(test_rmse_scores):.2f} (+/- {np.std(test_rmse_scores):.2f})")
        print(f"   Min: {np.min(test_rmse_scores):.2f}, Max: {np.max(test_rmse_scores):.2f}")
        
        # Analyse de la stabilite du modele
        print("\nANALYSE DE LA STABILITE:")
        print("-"*40)
        r2_std = np.std(test_r2_scores)
        if r2_std < 0.05:
            print("   Modele TRES STABLE (faible variance entre les plis)")
        elif r2_std < 0.1:
            print("   Modele STABLE (variance moderee entre les plis)")
        elif r2_std < 0.2:
            print("   Modele MODEREMENT STABLE (variance importante entre les plis)")
        else:
            print("   Modele INSTABLE (forte variance entre les plis)")
            print("   Suggestion: Augmenter le nombre de donnees ou regulariser le modele")
        
        # Verifier la difference entre train et test
        diff_train_test = np.mean(train_r2_scores) - np.mean(test_r2_scores)
        if diff_train_test > 0.2:
            print(f"\n   ATTENTION: Sur-apprentissage detecte (ecart Train-Test: {diff_train_test:.4f})")
            print("   Suggestion: Reduire la complexite du modele")
        elif diff_train_test > 0.1:
            print(f"\n   Sur-apprentissage modere (ecart Train-Test: {diff_train_test:.4f})")
        else:
            print(f"\n   Bonne generalisation (ecart Train-Test: {diff_train_test:.4f})")
        
        print("="*60)
        
        # Sauvegarder les resultats
        results = {
            'train_r2': train_r2_scores,
            'test_r2': test_r2_scores,
            'test_mae': test_mae_scores,
            'test_rmse': test_rmse_scores,
            'mean_train_r2': np.mean(train_r2_scores),
            'std_train_r2': np.std(train_r2_scores),
            'mean_test_r2': np.mean(test_r2_scores),
            'std_test_r2': np.std(test_r2_scores),
            'mean_test_mae': np.mean(test_mae_scores),
            'std_test_mae': np.std(test_mae_scores),
            'mean_test_rmse': np.mean(test_rmse_scores),
            'std_test_rmse': np.std(test_rmse_scores),
            'cv': cv
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation croisee: {e}")
        print(f"   Erreur: {e}")
        return None

def evaluer_modeles_avec_cv(X, y, cv=NOMBRE_PLIS_CV):
    """
    Compare les differents modeles avec validation croisee
    """
    print("\n" + "="*60)
    print("COMPARAISON DES MODELES AVEC VALIDATION CROISEE")
    print("="*60)
    
    # Preparer les donnees
    scaler_cv = StandardScaler()
    X_scaled = scaler_cv.fit_transform(X)
    
    # Modeles a comparer
    modeles = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
        'Hybride (poids 0.6/0.4)': HybridRegressor(
            rf_weight=0.6, knn_weight=0.4,
            rf_n_estimators=200, rf_max_depth=20,
            rf_min_samples_split=5, rf_min_samples_leaf=2,
            knn_n_neighbors=5, knn_weights='distance'
        )
    }
    
    # Definir les metriques
    scorers = {
        'r2': make_scorer(r2_score),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False)
    }
    
    results = {}
    
    for nom, modele in modeles.items():
        print(f"\nMODEL: {nom}")
        print("-"*40)
        
        try:
            cv_results = cross_validate(
                modele, X_scaled, y,
                cv=cv,
                scoring=scorers,
                n_jobs=-1,
                return_train_score=True,
                verbose=0
            )
            
            train_r2 = np.mean(cv_results['train_r2'])
            test_r2 = np.mean(cv_results['test_r2'])
            test_mae = -np.mean(cv_results['test_mae'])
            
            print(f"   R2 Train: {train_r2:.4f}")
            print(f"   R2 Test: {test_r2:.4f}")
            print(f"   MAE Test: {test_mae:.2f} Ohm.m")
            
            results[nom] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae
            }
            
        except Exception as e:
            print(f"   Erreur: {e}")
            results[nom] = None
    
    # Afficher le resume de la comparaison
    print("\n" + "="*60)
    print("RESUME DE LA COMPARAISON")
    print("="*60)
    print("\nPerformance sur le test set:")
    print("-"*40)
    
    for nom, res in results.items():
        if res is not None:
            print(f"   {nom:20s}: R2={res['test_r2']:.4f}, MAE={res['test_mae']:.2f} Ohm.m")
    
    # Identifier le meilleur modele
    meilleur_modele = max(results.items(), key=lambda x: x[1]['test_r2'] if x[1] is not None else -np.inf)
    print(f"\nMeilleur modele: {meilleur_modele[0]} (R2={meilleur_modele[1]['test_r2']:.4f})")
    print("="*60)
    
    return results

def recherche_hyperparametres_avec_cv(X, y, cv=NOMBRE_PLIS_CV):
    """
    Recherche des meilleurs hyperparametres avec validation croisee
    """
    print("\n" + "="*60)
    print("RECHERCHE DES MEILLEURS HYPERPARAMETRES AVEC VALIDATION CROISEE")
    print("="*60)
    
    # Preparer les donnees
    scaler_cv = StandardScaler()
    X_scaled = scaler_cv.fit_transform(X)
    
    # Definition de la grille de recherche pour le Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("\nRECHERCHE POUR RANDOM FOREST:")
    print("-"*40)
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_rf = GridSearchCV(
        rf, param_grid_rf,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_rf.fit(X_scaled, y)
    
    print(f"   Meilleurs parametres: {grid_rf.best_params_}")
    print(f"   Meilleur score R2: {grid_rf.best_score_:.4f}")
    
    # Definition de la grille de recherche pour le KNN
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }
    
    print("\nRECHERCHE POUR KNN:")
    print("-"*40)
    
    knn = KNeighborsRegressor(n_jobs=-1)
    grid_knn = GridSearchCV(
        knn, param_grid_knn,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_knn.fit(X_scaled, y)
    
    print(f"   Meilleurs parametres: {grid_knn.best_params_}")
    print(f"   Meilleur score R2: {grid_knn.best_score_:.4f}")
    
    # Definition de la grille de recherche pour le modele hybride
    param_grid_hybrid = {
        'rf_weight': [0.3, 0.5, 0.7, 0.9],
        'rf_n_estimators': [100, 200],
        'rf_max_depth': [15, 20, 25],
        'knn_n_neighbors': [3, 5, 7]
    }
    
    print("\nRECHERCHE POUR MODELE HYBRIDE:")
    print("-"*40)
    
    hybrid = HybridRegressor()
    grid_hybrid = GridSearchCV(
        hybrid, param_grid_hybrid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_hybrid.fit(X_scaled, y)
    
    print(f"   Meilleurs parametres: {grid_hybrid.best_params_}")
    print(f"   Meilleur score R2: {grid_hybrid.best_score_:.4f}")
    
    print("="*60)
    
    return {
        'rf': grid_rf.best_params_,
        'knn': grid_knn.best_params_,
        'hybrid': grid_hybrid.best_params_,
        'rf_score': grid_rf.best_score_,
        'knn_score': grid_knn.best_score_,
        'hybrid_score': grid_hybrid.best_score_
    }

# ================================
# FONCTIONS DE GESTION DU MODELE
# ================================

def sauvegarder_modele(model, scaler, fichier_modele=FICHIER_MODELE, fichier_scaler=FICHIER_SCALER):
    """Sauvegarde le modele et le scaler"""
    try:
        joblib.dump(model, fichier_modele)
        joblib.dump(scaler, fichier_scaler)
        logger.info(f"Modele sauvegarde : {fichier_modele}")
        logger.info(f"Scaler sauvegarde : {fichier_scaler}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modele : {e}")
        return False

def charger_modele(fichier_modele=FICHIER_MODELE, fichier_scaler=FICHIER_SCALER):
    """Charge le modele et le scaler s'ils existent"""
    try:
        if os.path.exists(fichier_modele) and os.path.exists(fichier_scaler):
            model = joblib.load(fichier_modele)
            scaler = joblib.load(fichier_scaler)
            logger.info(f"Modele charge : {fichier_modele}")
            return model, scaler
        return None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
        return None, None

def charger_train_test():
    """Charge les ensembles d'entraînement et de test sauvegardes"""
    try:
        if os.path.exists(FICHIER_TRAIN_TEST):
            data = joblib.load(FICHIER_TRAIN_TEST)
            return data.get('X_train'), data.get('X_test'), data.get('y_train'), data.get('y_test')
        return None, None, None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des ensembles train/test : {e}")
        return None, None, None, None

def evaluer_modele(model, scaler, X, y, nom="Modele"):
    """
    Evalue les performances du modele et retourne les metriques
    """
    try:
        Xs = scaler.transform(X)
        y_pred = model.predict(Xs)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Affichage formaté
        print(f"\n   EVALUATION DU {nom.upper()}:")
        print("   " + "-"*40)
        print(f"   MAE  (Erreur absolue moyenne)   : {mae:.2f} Ohm.m")
        print(f"   RMSE (Racine de l'erreur quad.)  : {rmse:.2f} Ohm.m")
        print(f"   R2   (Coefficient determination) : {r2:.4f}")
        print("   " + "-"*40)
        
        # Interpretation du R2
        if r2 >= 0.9:
            print("   Performance : EXCELLENTE")
        elif r2 >= 0.8:
            print("   Performance : TRES BONNE")
        elif r2 >= 0.7:
            print("   Performance : BONNE")
        elif r2 >= 0.5:
            print("   Performance : MOYENNE")
        else:
            print("   Performance : INSUFFISANTE")
        
        return mae, rmse, r2
    except Exception as e:
        logger.error(f"Erreur lors de l'evaluation du modele : {e}")
        return None, None, None

def afficher_metriques_comparatives(model, scaler, X_train, y_train, X_test, y_test):
    """
    Affiche les metriques comparatives sur les ensembles d'entraînement et de test
    """
    print("\n" + "="*60)
    print("METRIQUES DE PERFORMANCE COMPLETES")
    print("="*60)
    
    # Metriques sur l'ensemble d'entraînement
    print("\nPERFORMANCE SUR L'ENSEMBLE D'ENTRAINEMENT:")
    mae_train, rmse_train, r2_train = evaluer_modele(model, scaler, X_train, y_train, "Modele sur Train")
    
    # Metriques sur l'ensemble de test
    print("\nPERFORMANCE SUR L'ENSEMBLE DE TEST:")
    mae_test, rmse_test, r2_test = evaluer_modele(model, scaler, X_test, y_test, "Modele sur Test")
    
    if all(v is not None for v in [mae_train, rmse_train, r2_train, mae_test, rmse_test, r2_test]):
        # Comparaison
        print("\nCOMPARAISON TRAIN VS TEST:")
        print("   " + "-"*40)
        print(f"   R2  : Train={r2_train:.4f} | Test={r2_test:.4f} | Difference={abs(r2_train-r2_test):.4f}")
        print(f"   MAE : Train={mae_train:.2f} | Test={mae_test:.2f} | Difference={abs(mae_train-mae_test):.2f}")
        
        # Detection de sur-apprentissage
        if abs(r2_train - r2_test) > 0.1:
            print("   ATTENTION : Ecart important entre Train et Test (risque de sur-apprentissage)")
        elif r2_test > 0.7:
            print("   Bonne generalisation du modele")
        else:
            print("   Performance a ameliorer sur les donnees de test")
    
    print("="*60)
    
    return {
        'train': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train},
        'test': {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test}
    }

def evaluer_modeles_separement(rf_model, knn_model, scaler, X_train, y_train, X_test, y_test):
    """
    Evalue les performances des modeles individuels
    """
    print("\n" + "="*60)
    print("EVALUATION DES MODELES INDIVIDUELS")
    print("="*60)
    
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)
    
    # Evaluation du Random Forest
    print("\nPERFORMANCE DU RANDOM FOREST:")
    rf_train_pred = rf_model.predict(Xs_train)
    rf_test_pred = rf_model.predict(Xs_test)
    
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
    rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
    
    print(f"   Train R2: {rf_train_r2:.4f}, MAE: {rf_train_mae:.2f}")
    print(f"   Test R2: {rf_test_r2:.4f}, MAE: {rf_test_mae:.2f}")
    
    # Evaluation du KNN
    print("\nPERFORMANCE DU KNN:")
    knn_train_pred = knn_model.predict(Xs_train)
    knn_test_pred = knn_model.predict(Xs_test)
    
    knn_train_r2 = r2_score(y_train, knn_train_pred)
    knn_test_r2 = r2_score(y_test, knn_test_pred)
    knn_train_mae = mean_absolute_error(y_train, knn_train_pred)
    knn_test_mae = mean_absolute_error(y_test, knn_test_pred)
    
    print(f"   Train R2: {knn_train_r2:.4f}, MAE: {knn_train_mae:.2f}")
    print(f"   Test R2: {knn_test_r2:.4f}, MAE: {knn_test_mae:.2f}")
    
    # Comparaison
    print("\nCOMPARAISON DES MODELES:")
    print("-"*40)
    print(f"   RF - Test R2: {rf_test_r2:.4f}")
    print(f"   KNN - Test R2: {knn_test_r2:.4f}")
    print(f"   Difference: {abs(rf_test_r2 - knn_test_r2):.4f}")
    
    if rf_test_r2 > knn_test_r2:
        print("   Le Random Forest performe mieux que le KNN")
    elif knn_test_r2 > rf_test_r2:
        print("   Le KNN performe mieux que le Random Forest")
    else:
        print("   Les deux modeles ont des performances similaires")
    
    print("="*60)
    
    return {
        'rf': {'train_r2': rf_train_r2, 'test_r2': rf_test_r2, 'train_mae': rf_train_mae, 'test_mae': rf_test_mae},
        'knn': {'train_r2': knn_train_r2, 'test_r2': knn_test_r2, 'train_mae': knn_train_mae, 'test_mae': knn_test_mae}
    }

def optimiser_poids_hybride(rf_model, knn_model, scaler, X_train, y_train, X_test, y_test):
    """
    Optimise les poids du modele hybride par recherche sur grille
    """
    print("\n" + "="*60)
    print("OPTIMISATION DES POIDS DU MODELE HYBRIDE")
    print("="*60)
    
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)
    
    rf_pred = rf_model.predict(Xs_train)
    knn_pred = knn_model.predict(Xs_train)
    
    # Test differents poids
    meilleur_r2 = -np.inf
    meilleur_poids = 0.5
    
    for rf_weight in np.arange(0, 1.1, 0.1):
        knn_weight = 1 - rf_weight
        hybrid_pred = rf_weight * rf_pred + knn_weight * knn_pred
        r2 = r2_score(y_train, hybrid_pred)
        
        if r2 > meilleur_r2:
            meilleur_r2 = r2
            meilleur_poids = rf_weight
    
    print(f"   Poids optimal RF: {meilleur_poids:.1f}")
    print(f"   Poids optimal KNN: {1-meilleur_poids:.1f}")
    print(f"   R2 sur l'entraînement: {meilleur_r2:.4f}")
    
    # Evaluation sur le test set avec les poids optimises
    rf_test_pred = rf_model.predict(Xs_test)
    knn_test_pred = knn_model.predict(Xs_test)
    hybrid_test_pred = meilleur_poids * rf_test_pred + (1 - meilleur_poids) * knn_test_pred
    r2_test = r2_score(y_test, hybrid_test_pred)
    
    print(f"   R2 sur le test set: {r2_test:.4f}")
    print("="*60)
    
    return meilleur_poids

def reentrainement_automatique(data_train, model, scaler, historique, 
                               data_initial_complete=None, seuil=SEUIL_REENTRAINEMENT):
    """
    Fonction de reentraînement automatique du modele
    Version corrigee avec gestion correcte de l'historique
    """
    print("\n" + "="*60)
    print("VERIFICATION DU REENTRAINEMENT AUTOMATIQUE")
    print("="*60)
    
    if data_train is None:
        print("   Erreur: Donnees d'entraînement non disponibles")
        return model, scaler, historique
    
    nb_nouveaux = len(historique)
    print(f"   {nb_nouveaux} nouveaux points a integrer")
    
    besoin_reentrainement = False
    raison = []
    
    # Seuil base sur le nombre de points dans l'historique
    if nb_nouveaux >= seuil:
        besoin_reentrainement = True
        raison.append(f"Nombre de nouveaux points ({nb_nouveaux}) >= seuil ({seuil})")
    
    if model is not None and scaler is not None:
        try:
            # Evaluation sur les donnees d'entraînement uniquement
            X_train = data_train[["latitude", "longitude", "altitude"]]
            y_train = data_train["wenner"]
            _, _, r2 = evaluer_modele(model, scaler, X_train, y_train, "modele actuel")
            if r2 is not None and r2 < SEUIL_PERFORMANCE:
                besoin_reentrainement = True
                raison.append(f"Performance R2 ({r2:.3f}) < seuil ({SEUIL_PERFORMANCE})")
        except Exception as e:
            logger.error(f"Erreur lors de l'evaluation du modele: {e}")
            besoin_reentrainement = True
            raison.append("Erreur lors de l'evaluation du modele")
    
    if besoin_reentrainement:
        print("\n   LANCEMENT DU REENTRAINEMENT...")
        if raison:
            for r in raison:
                print(f"      {r}")
        
        # Utiliser les donnees d'entraînement initiales si non fournies
        if data_initial_complete is None:
            # Charger les donnees initiales si elles existent
            if os.path.exists(FICHIER_HISTORIQUE_INITIAL):
                data_initial_complete = pd.read_csv(FICHIER_HISTORIQUE_INITIAL)
            else:
                # Sinon utiliser les donnees d'entraînement actuelles
                data_initial_complete = data_train[["latitude", "longitude", "altitude", "wenner"]].copy()
                # Sauvegarder les donnees initiales pour reference future
                data_initial_complete.to_csv(FICHIER_HISTORIQUE_INITIAL, index=False)
        
        # Fusionner les donnees initiales avec l'historique
        data_complete = pd.concat([
            data_initial_complete[["latitude", "longitude", "altitude", "wenner"]], 
            historique
        ], ignore_index=True)
        
        X_complete = data_complete[["latitude", "longitude", "altitude"]]
        y_complete = data_complete["wenner"]
        
        scaler_nouveau = StandardScaler()
        Xs_complete = scaler_nouveau.fit_transform(X_complete)
        
        # Creation du modele hybride avec meilleurs parametres
        model_nouveau = HybridRegressor(
            rf_weight=POIDS_RF,
            knn_weight=POIDS_KNN,
            rf_n_estimators=100,      # Meilleur parametre
            rf_max_depth=10,          # Meilleur parametre (reduit pour eviter sur-apprentissage)
            rf_min_samples_split=10,  # Meilleur parametre
            rf_min_samples_leaf=4,    # Meilleur parametre
            knn_n_neighbors=11,       # Meilleur parametre
            knn_weights='uniform'     # Meilleur parametre
        )
        
        print("   Entraînement du nouveau modele hybride en cours...")
        start_time = time.time()
        model_nouveau.fit(Xs_complete, y_complete)
        end_time = time.time()
        
        print(f"   Reentraînement termine en {end_time - start_time:.1f} secondes")
        print(f"   Nouveau nombre de points : {len(data_complete)}")
        
        # Evaluation du nouveau modele
        evaluer_modele(model_nouveau, scaler_nouveau, X_complete, y_complete, "nouveau modele hybride")
        
        # Validation croisee du nouveau modele
        evaluer_validation_croisee(model_nouveau, Xs_complete, y_complete)
        
        sauvegarder_modele(model_nouveau, scaler_nouveau)
        
        # Sauvegarder UNIQUEMENT l'historique (pas les donnees initiales)
        # Reinitialiser l'historique pour le prochain cycle
        historique_vide = pd.DataFrame(columns=["latitude", "longitude", "altitude", "wenner"])
        historique_vide.to_csv(FICHIER_HISTORIQUE, index=False)
        
        print("   Historique reinitialise apres reentraînement")
        
        return model_nouveau, scaler_nouveau, pd.DataFrame()
        
    else:
        print("   Aucun reentraînement necessaire")
        print(f"   Prochain reentraînement apres {seuil - nb_nouveaux} nouveaux points")
        historique.to_csv(FICHIER_HISTORIQUE, index=False)
        return model, scaler, historique

def reentrainement_periodique(data_train, model, scaler, intervalle_jours=INTERVALLE_ENTRAINEMENT_DEFAUT):
    """Reentraînement periodique du modele"""
    try:
        if data_train is None:
            logger.warning("Donnees d'entraînement non disponibles pour le reentraînement periodique")
            return model, scaler
        
        if os.path.exists(FICHIER_DERNIER_ENTRAINEMENT):
            with open(FICHIER_DERNIER_ENTRAINEMENT, 'r') as f:
                date_dernier = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        else:
            date_dernier = datetime.now() - pd.Timedelta(days=intervalle_jours + 1)
        
        diff_jours = (datetime.now() - date_dernier).days
        
        if diff_jours >= intervalle_jours:
            print(f"\nREENTRAINEMENT PERIODIQUE (apres {diff_jours} jours)")
            
            data_complete = data_train[["latitude", "longitude", "altitude", "wenner"]].copy()
            
            if os.path.exists(FICHIER_HISTORIQUE):
                hist = pd.read_csv(FICHIER_HISTORIQUE)
                if not hist.empty:
                    data_complete = pd.concat([data_complete, hist], ignore_index=True)
            
            X_complete = data_complete[["latitude", "longitude", "altitude"]]
            y_complete = data_complete["wenner"]
            
            scaler_nouveau = StandardScaler()
            Xs_complete = scaler_nouveau.fit_transform(X_complete)
            
            # Creation du modele hybride avec meilleurs parametres
            model_nouveau = HybridRegressor(
                rf_weight=POIDS_RF,
                knn_weight=POIDS_KNN,
                rf_n_estimators=100,
                rf_max_depth=10,
                rf_min_samples_split=10,
                rf_min_samples_leaf=4,
                knn_n_neighbors=11,
                knn_weights='uniform'
            )
            
            print(f"   Entraînement sur {len(data_complete)} points...")
            model_nouveau.fit(Xs_complete, y_complete)
            
            # Validation croisee
            evaluer_validation_croisee(model_nouveau, Xs_complete, y_complete)
            
            sauvegarder_modele(model_nouveau, scaler_nouveau)
            
            with open(FICHIER_DERNIER_ENTRAINEMENT, 'w') as f:
                f.write(datetime.now().strftime('%Y-%m-%d'))
            
            print("   Reentraînement periodique termine")
            return model_nouveau, scaler_nouveau
        
        return model, scaler
    except Exception as e:
        logger.error(f"Erreur lors du reentraînement periodique: {e}")
        return model, scaler

def nettoyer_historique():
    """
    Nettoie l'historique pour eviter l'accumulation excessive
    """
    try:
        if os.path.exists(FICHIER_HISTORIQUE):
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            if len(historique) > TAILLE_MAX_HISTORIQUE:
                print(f"   Nettoyage de l'historique ({len(historique)} points)")
                # Garder seulement les derniers points
                historique_recent = historique.tail(TAILLE_MAX_HISTORIQUE // 2)
                historique_recent.to_csv(FICHIER_HISTORIQUE, index=False)
                print(f"   Historique reduit a {len(historique_recent)} points")
                return historique_recent
        return None
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage de l'historique: {e}")
        return None

def analyser_importance_features(model, X):
    """Analyse l'importance des caracteristiques pour le modele hybride"""
    if hasattr(model, 'get_feature_importances'):
        importances = model.get_feature_importances()
        features = X.columns
        
        print("\nIMPORTANCE DES CARACTERISTIQUES (MODELE HYBRIDE):")
        print("-"*40)
        for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
            barre = "|" * int(imp * 30)
            print(f"   {feat:12s} : {imp:.3f} ({imp*100:.1f}%) {barre}")
        print("-"*40)
        
        return dict(zip(features, importances))
    return None

def valider_coordonnees(lat, lon, alt):
    """Valide les coordonnees saisies"""
    try:
        lat = float(lat)
        lon = float(lon)
        alt = float(alt)
        
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude invalide: {lat}. Doit etre entre -90 et 90")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude invalide: {lon}. Doit etre entre -180 et 180")
        if alt < -500 or alt > 9000:
            raise ValueError(f"Altitude invalide: {alt}. Doit etre entre -500 et 9000m")
        
        return lat, lon, alt
    except ValueError as e:
        raise ValueError(f"Erreur de validation: {e}")

def charger_donnees_csv(fichier):
    """Charge et nettoie les donnees CSV"""
    try:
        data = pd.read_csv(fichier, sep=";")
        
        colonnes_requises = ["latitude", "longitude", "altitude", "wenner"]
        for col in colonnes_requises:
            if col not in data.columns:
                raise ValueError(f"Colonne '{col}' manquante dans le fichier")
        
        for col in colonnes_requises:
            try:
                data[col] = data[col].astype(str).str.replace(",", ".").astype(float)
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Erreur de conversion pour la colonne {col}: {e}")
        
        # Supprimer les lignes avec des valeurs manquantes
        data = data.dropna()
        
        # Verifier qu'il reste des donnees
        if len(data) == 0:
            raise ValueError("Aucune donnee valide apres nettoyage")
        
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement des donnees: {e}")
        raise

def analyser_donnees(data):
    """Analyse descriptive des donnees"""
    print("\n" + "="*60)
    print("ANALYSE DESCRIPTIVE DES DONNEES")
    print("="*60)
    print(f"   Nombre de points: {len(data)}")
    print(f"   Resistivite - Min: {data['wenner'].min():.2f} Ohm.m")
    print(f"   Resistivite - Max: {data['wenner'].max():.2f} Ohm.m")
    print(f"   Resistivite - Moyenne: {data['wenner'].mean():.2f} Ohm.m")
    print(f"   Resistivite - Ecart-type: {data['wenner'].std():.2f} Ohm.m")
    print("\n   Correlations avec la resistivite:")
    print(f"      Latitude: {data['latitude'].corr(data['wenner']):.4f}")
    print(f"      Longitude: {data['longitude'].corr(data['wenner']):.4f}")
    print(f"      Altitude: {data['altitude'].corr(data['wenner']):.4f}")
    print("="*60)

# ================================
# 1. CHARGEMENT DES DONNEES
# ================================
print("\n1. CHARGEMENT DES DONNEES...")

if not os.path.exists(FICHIER_DONNEES):
    print(f"Erreur : Le fichier {FICHIER_DONNEES} n'existe pas.")
    print("   Creez un fichier CSV avec les colonnes : latitude;longitude;altitude;wenner")
    exit()

try:
    data = charger_donnees_csv(FICHIER_DONNEES)
    print(f"{len(data)} points charges depuis {FICHIER_DONNEES}")
    
    # Analyse descriptive des donnees
    analyser_donnees(data)
except Exception as e:
    print(f"Erreur lors du chargement des donnees: {e}")
    exit()

# ================================
# 2. ANALYSE PRELIMINAIRE AVEC VALIDATION CROISEE
# ================================
print("\n2. ANALYSE PRELIMINAIRE AVEC VALIDATION CROISEE...")

X = data[["latitude", "longitude", "altitude"]]
y = data["wenner"]

# Comparaison des modeles avec validation croisee
resultats_comparaison = evaluer_modeles_avec_cv(X, y)

# Recherche des meilleurs hyperparametres
meilleurs_params = recherche_hyperparametres_avec_cv(X, y)

# ================================
# 3. CHARGEMENT OU CREATION DU MODELE
# ================================
print("\n3. CHARGEMENT/CREATION DU MODELE HYBRIDE...")

# Essayer de charger un modele existant
model, scaler = charger_modele()

# Charger les ensembles train/test existants
X_train_global, X_test_global, y_train_global, y_test_global = charger_train_test()

if model is None or scaler is None:
    print("   Creation d'un nouveau modele hybride...")
    
    # Division en train/test pour evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Sauvegarder les ensembles pour usage futur
    X_train_global = X_train
    X_test_global = X_test
    y_train_global = y_train
    y_test_global = y_test
    
    # Sauvegarder les ensembles de test pour reference future
    train_test_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    joblib.dump(train_test_data, FICHIER_TRAIN_TEST)
    
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)
    
    # Utiliser les meilleurs parametres trouves ou les valeurs par defaut
    if meilleurs_params and 'hybrid' in meilleurs_params:
        print("   Utilisation des parametres optimises par validation croisee...")
        model = HybridRegressor(
            rf_weight=meilleurs_params['hybrid'].get('rf_weight', POIDS_RF),
            knn_weight=1-meilleurs_params['hybrid'].get('rf_weight', POIDS_RF),
            rf_n_estimators=meilleurs_params['hybrid'].get('rf_n_estimators', 100),
            rf_max_depth=meilleurs_params['hybrid'].get('rf_max_depth', 10),
            rf_min_samples_split=meilleurs_params['hybrid'].get('rf_min_samples_split', 10),
            rf_min_samples_leaf=meilleurs_params['hybrid'].get('rf_min_samples_leaf', 4),
            knn_n_neighbors=meilleurs_params['hybrid'].get('knn_n_neighbors', 11),
            knn_weights='uniform'
        )
    else:
        print("   Utilisation des parametres par defaut optimises...")
        model = HybridRegressor(
            rf_weight=POIDS_RF,
            knn_weight=POIDS_KNN,
            rf_n_estimators=100,
            rf_max_depth=10,
            rf_min_samples_split=10,
            rf_min_samples_leaf=4,
            knn_n_neighbors=11,
            knn_weights='uniform'
        )
    
    print("   Entraînement du modele hybride...")
    model.fit(Xs_train, y_train)
    
    # Sauvegarder le modele
    sauvegarder_modele(model, scaler)
    
    # AFFICHER LES METRIQUES DE PERFORMANCE
    print("\n" + "="*60)
    print("ANALYSE DES PERFORMANCES DU MODELE HYBRIDE INITIAL")
    print("="*60)
    
    # Evaluation du modele hybride
    print("\nPERFORMANCE DU MODELE HYBRIDE:")
    print("\nPERFORMANCE SUR L'ENSEMBLE D'ENTRAINEMENT:")
    mae_train, rmse_train, r2_train = evaluer_modele(model, scaler, X_train, y_train, "Modele Hybride (Train)")
    
    print("\nPERFORMANCE SUR L'ENSEMBLE DE TEST:")
    mae_test, rmse_test, r2_test = evaluer_modele(model, scaler, X_test, y_test, "Modele Hybride (Test)")
    
    # Validation croisee du modele hybride
    cv_results = evaluer_validation_croisee(model, Xs_train, y_train)
    
    # Evaluation des modeles individuels
    rf_model = model.rf_model
    knn_model = model.knn_model
    evaluer_modeles_separement(rf_model, knn_model, scaler, X_train, y_train, X_test, y_test)
    
    # Optimisation des poids
    poids_optimal = optimiser_poids_hybride(rf_model, knn_model, scaler, X_train, y_train, X_test, y_test)
    
    # Recreer le modele avec les poids optimises
    if abs(poids_optimal - POIDS_RF) > 0.05:
        print("\nRECREATION DU MODELE AVEC LES POIDS OPTIMISES...")
        model_optimise = HybridRegressor(
            rf_weight=poids_optimal,
            knn_weight=1-poids_optimal,
            rf_n_estimators=100,
            rf_max_depth=10,
            rf_min_samples_split=10,
            rf_min_samples_leaf=4,
            knn_n_neighbors=11,
            knn_weights='uniform'
        )
        model_optimise.fit(Xs_train, y_train)
        model = model_optimise
        sauvegarder_modele(model, scaler)
        
        print("\nPERFORMANCE DU MODELE HYBRIDE OPTIMISE:")
        evaluer_modele(model, scaler, X_test, y_test, "Modele Hybride Optimise (Test)")
        
        # Validation croisee du modele optimise
        evaluer_validation_croisee(model, Xs_train, y_train)
    
    # Sauvegarder les donnees initiales pour le reentraînement futur
    data_initial = X_train.copy()
    data_initial["wenner"] = y_train
    data_initial.to_csv(FICHIER_HISTORIQUE_INITIAL, index=False)
    
    # Comparaison detaillee
    print("\n" + "="*60)
    print("COMPARAISON DETAILLEE TRAIN vs TEST (MODELE HYBRIDE)")
    print("="*60)
    print(f"   R2  : Train={r2_train:.4f} | Test={r2_test:.4f} | Ecart={abs(r2_train-r2_test):.4f}")
    print(f"   RMSE: Train={rmse_train:.2f} | Test={rmse_test:.2f} | Ecart={abs(rmse_train-rmse_test):.2f}")
    print(f"   MAE : Train={mae_train:.2f} | Test={mae_test:.2f} | Ecart={abs(mae_train-mae_test):.2f}")
    print("   " + "-"*40)
    
    # Diagnostic du modele
    if r2_test > 0.8 and abs(r2_train - r2_test) < 0.1:
        print("   MODELE PERFORMANT : Bonne generalisation et faible ecart Train/Test")
    elif r2_test > 0.7 and abs(r2_train - r2_test) < 0.15:
        print("   MODELE CORRECT : Performance acceptable")
    elif abs(r2_train - r2_test) > 0.15:
        print("   SUR-APPRENTISSAGE DETECTE : Ecart important entre Train et Test")
        print("      Suggestion : Reduire max_depth ou augmenter min_samples_split")
    else:
        print("   MODELE A AMELIORER : Performance insuffisante sur le test set")
    
    print("="*60)
    
    # Sauvegarder les resultats de la validation croisee
    if cv_results:
        joblib.dump(cv_results, FICHIER_CV_RESULTS)
else:
    print("   Modele hybride existant charge avec succes")
    # Si les ensembles train/test ne sont pas charges, les recreer
    if X_train_global is None:
        X = data[["latitude", "longitude", "altitude"]]
        y = data["wenner"]
        X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

# Analyser l'importance des caracteristiques
if X_train_global is not None:
    analyser_importance_features(model, X_train_global)

# ================================
# 4. CHOIX DU MODE DE PREDICTION
# ================================
print("\n4. SAISIE DES COORDONNEES")
print("-"*40)
print("1 - Saisie manuelle")
print("2 - Import depuis fichier CSV")

mode = input("Votre choix (1 ou 2) : ")

points = None

if mode == "2":
    fichier = input("Nom du fichier CSV a importer : ")
    
    if not os.path.exists(fichier):
        print(f"Fichier {fichier} introuvable")
        exit()
    
    try:
        points = pd.read_csv(fichier, sep=";")
        
        colonnes_requises = ["latitude", "longitude", "altitude"]
        for col in colonnes_requises:
            if col not in points.columns:
                print(f"Colonne '{col}' manquante dans le fichier")
                exit()
        
        for col in colonnes_requises:
            points[col] = points[col].astype(str).str.replace(",", ".").astype(float)
        
        print(f"{len(points)} points charges depuis {fichier}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        exit()

else:
    lignes = []
    
    while True:
        print("\nNOUVEAU POINT")
        try:
            lat_input = input("   Latitude : ")
            lon_input = input("   Longitude : ")
            alt_input = input("   Altitude (m) : ")
            
            lat, lon, alt = valider_coordonnees(lat_input, lon_input, alt_input)
            
            lignes.append([lat, lon, alt])
            
            rep = input("\nAjouter un autre point ? (o/n) : ")
            if rep.lower() != "o":
                break
                
        except ValueError as e:
            print(f"   Erreur : {e}")
    
    points = pd.DataFrame(lignes, columns=["latitude", "longitude", "altitude"])
    print(f"\n{len(points)} points saisis")

# ================================
# 5. PREDICTION
# ================================
print("\n5. PREDICTION EN COURS...")

try:
    Xp = scaler.transform(points)
    points["wenner"] = model.predict(Xp)
    
    print("\nRESULTATS DES PREDICTIONS :")
    print("="*50)
    print(points.to_string(index=False))
    print("="*50)
except Exception as e:
    print(f"Erreur lors de la prediction: {e}")
    exit()

# ================================
# 6. GESTION DE L'HISTORIQUE
# ================================
print("\n6. GESTION DE L'HISTORIQUE...")

# Nettoyer l'historique avant de l'utiliser
historique_nettoye = nettoyer_historique()

try:
    if os.path.exists(FICHIER_HISTORIQUE):
        ancien = pd.read_csv(FICHIER_HISTORIQUE)
        # Verifier que l'historique a le bon format
        colonnes_requises = ["latitude", "longitude", "altitude", "wenner"]
        if all(col in ancien.columns for col in colonnes_requises):
            # Limiter la taille de l'historique
            if len(ancien) > TAILLE_MAX_HISTORIQUE:
                print(f"   Historique trop grand ({len(ancien)} points), conservation des {TAILLE_MAX_HISTORIQUE//2} derniers")
                ancien = ancien.tail(TAILLE_MAX_HISTORIQUE // 2)
            historique = pd.concat([ancien, points], ignore_index=True)
        else:
            print("   Format d'historique incorrect, reinitialisation...")
            historique = points.copy()
    else:
        historique = points.copy()
    
    # Limiter la taille de l'historique
    if len(historique) > TAILLE_MAX_HISTORIQUE:
        print(f"   Historique limite a {TAILLE_MAX_HISTORIQUE} points (actuellement {len(historique)})")
        historique = historique.tail(TAILLE_MAX_HISTORIQUE)
    
    historique.to_csv(FICHIER_HISTORIQUE, index=False)
    print(f"{len(points)} points ajoutes a l'historique")
    print(f"Total dans l'historique : {len(historique)} points")
except Exception as e:
    logger.error(f"Erreur lors de la gestion de l'historique: {e}")
    print(f"   Erreur: {e}")
    historique = points.copy()
    historique.to_csv(FICHIER_HISTORIQUE, index=False)

# ================================
# 7. REENTRAINEMENT AUTOMATIQUE
# ================================
print("\n7. REENTRAINEMENT AUTOMATIQUE...")
try:
    if X_train_global is not None:
        data_train = X_train_global.copy()
        data_train["wenner"] = y_train_global
        
        # Charger les donnees initiales sauvegardees
        data_initial = None
        if os.path.exists(FICHIER_HISTORIQUE_INITIAL):
            data_initial = pd.read_csv(FICHIER_HISTORIQUE_INITIAL)
        
        model, scaler, historique = reentrainement_automatique(
            data_train, model, scaler, historique, data_initial, SEUIL_REENTRAINEMENT
        )
    else:
        print("   Donnees d'entraînement non disponibles, reentraînement annule")
except Exception as e:
    logger.error(f"Erreur lors du reentraînement automatique: {e}")
    print(f"   Erreur lors du reentraînement automatique: {e}")
    print("   Le modele actuel est conserve")

# ================================
# 7.2 REENTRAINEMENT PERIODIQUE (optionnel)
# ================================
print("\n7.2 REENTRAINEMENT PERIODIQUE (optionnel)")
try:
    reentrainement_periodique_choice = input("   Activer le reentraînement periodique ? (o/n) : ")
    if reentrainement_periodique_choice.lower() == 'o':
        intervalle_input = input("   Intervalle en jours (defaut: 7) : ")
        
        if intervalle_input.strip() == '':
            intervalle_jours = 7
        else:
            try:
                intervalle_jours = int(intervalle_input)
                if intervalle_jours <= 0:
                    print("   L'intervalle doit etre positif, utilisation de 7 jours")
                    intervalle_jours = 7
            except ValueError:
                print("   Valeur invalide, utilisation de 7 jours par defaut")
                intervalle_jours = 7
        
        print(f"   Intervalle configure : {intervalle_jours} jours")
        
        # Preparer les donnees d'entraînement
        if X_train_global is not None:
            data_train = X_train_global.copy()
            data_train["wenner"] = y_train_global
            model, scaler = reentrainement_periodique(data_train, model, scaler, intervalle_jours)
        else:
            print("   Donnees d'entraînement non disponibles, reentraînement annule")
    else:
        print("   Reentraînement periodique desactive")
except Exception as e:
    logger.error(f"Erreur lors du reentraînement periodique: {e}")
    print(f"   Erreur lors du reentraînement periodique: {e}")
    print("   Le modele actuel est conserve")

# ================================
# 8. CARTE INTERACTIVE
# ================================
print("\n8. GENERATION DE LA CARTE...")

try:
    centre = [points["latitude"].mean(), points["longitude"].mean()]

    m = folium.Map(
        location=centre,
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )

    for i, row in points.iterrows():
        folium.Circle(
            [row["latitude"], row["longitude"]],
            radius=100,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.1,
            weight=1,
            popup=f"Zone +/-100m - Resistivite: {row['wenner']:.1f} Ohm.m"
        ).add_to(m)
        
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=f"""
            <b>Resistivite Wenner</b><br>
            <b style="font-size:16px">{row['wenner']:.1f} Ohm.m</b><br>
            <hr>
            Lat: {row['latitude']:.6f}<br>
            Lon: {row['longitude']:.6f}<br>
            Alt: {row['altitude']:.1f} m
            """,
            tooltip=f"{row['wenner']:.1f} Ohm.m",
            icon=folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
        ).add_to(m)

    plugins.MiniMap().add_to(m)
    folium.LayerControl().add_to(m)

    m.save("carte_resistivite_wenner.html")
    print("Carte sauvegardee : carte_resistivite_wenner.html")
except Exception as e:
    logger.error(f"Erreur lors de la generation de la carte: {e}")
    print(f"Erreur lors de la generation de la carte: {e}")

# ================================
# 9. RECAPITULATIF DETAILLE
# ================================
print("\n" + "="*60)
print("RECAPITULATIF DES PREDICTIONS")
print("="*60)
print(f"Nombre de points predits : {len(points)}")
print(f"Resistivite moyenne      : {points['wenner'].mean():.1f} Ohm.m")
print(f"Resistivite min          : {points['wenner'].min():.1f} Ohm.m")
print(f"Resistivite max          : {points['wenner'].max():.1f} Ohm.m")
if len(points) > 1:
    print(f"Ecart-type               : {points['wenner'].std():.1f} Ohm.m")

print("\nINFORMATIONS SUR LE MODELE HYBRIDE:")
print(f"   Type: Hybrid Regressor (Random Forest + KNN)")
print(f"   Poids RF: {model.rf_weight:.2f}")
print(f"   Poids KNN: {model.knn_weight:.2f}")

if hasattr(model, 'rf_model'):
    print(f"\n   RANDOM FOREST:")
    print(f"      Nombre d'arbres: {model.rf_model.n_estimators}")
    print(f"      Profondeur max: {model.rf_model.max_depth}")

if hasattr(model, 'knn_model'):
    print(f"\n   KNN:")
    print(f"      Nombre de voisins: {model.knn_model.n_neighbors}")
    print(f"      Poids: {model.knn_model.weights}")

if hasattr(model, 'get_feature_importances') and X_train_global is not None:
    print(f"\n   Importance des variables (modele hybride):")
    importances = model.get_feature_importances()
    for feat, imp in zip(["latitude", "longitude", "altitude"], importances):
        barre = "|" * int(imp * 30)
        print(f"      {feat:10s}: {imp:.3f} {barre}")

# Charger les resultats de la validation croisee
cv_results = None
if os.path.exists(FICHIER_CV_RESULTS):
    try:
        cv_results = joblib.load(FICHIER_CV_RESULTS)
    except:
        pass

if cv_results:
    print("\nRESULTATS DE LA VALIDATION CROISEE:")
    print(f"   R2 moyen (cv): {cv_results['mean_test_r2']:.4f} (+/- {cv_results['std_test_r2']:.4f})")
    print(f"   MAE moyen (cv): {cv_results['mean_test_mae']:.2f} Ohm.m")
    print(f"   RMSE moyen (cv): {cv_results['mean_test_rmse']:.2f} Ohm.m")

print("\nFICHIERS GENERES:")
print(f"   - {FICHIER_MODELE} (modele hybride entraîne)")
print("   - scaler_wenner.pkl (normalisateur)")
print("   - historique_predictions_wenner.csv (historique des predictions)")
print("   - historique_initial.csv (donnees initiales pour reentraînement)")
print("   - carte_resistivite_wenner.html (carte interactive)")
print("   - train_test_split.pkl (ensembles d'entraînement et de test)")
print("   - dernier_entrainement.txt (date du dernier reentraînement periodique)")
print("   - cross_validation_results.pkl (resultats de la validation croisee)")

print("\n" + "="*60)
print("PROGRAMME TERMINE AVEC SUCCES")
print("="*60)