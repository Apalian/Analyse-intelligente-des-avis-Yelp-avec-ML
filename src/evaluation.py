import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(y_true, y_pred, model_name):
    """Évalue et sauvegarde les résultats d'un modèle."""
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"F1-macro: {results['f1_macro']:.3f}")
    print(classification_report(y_true, y_pred))
    
    return results
