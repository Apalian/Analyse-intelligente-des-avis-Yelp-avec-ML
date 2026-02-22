import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

RESULTS_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'all_results.csv')


def save_result(result, csv_path=RESULTS_CSV):
    """
    Sauvegarde un résultat dans le CSV. Si le model_name existe déjà, il est remplacé.

    Args:
        result: dict avec au minimum 'model', 'accuracy', 'f1_macro', 'f1_weighted'
        csv_path: chemin vers le fichier CSV
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['model'] != result['model']]
    else:
        df = pd.DataFrame()

    new_row = pd.DataFrame([result])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def evaluate_model(model, X_test, y_test, model_name, vectorizer=None, save=True):
    """
    Évalue un modèle sklearn et affiche les métriques.

    Args:
        model: Le modèle entraîné
        X_test: Données de test (texte ou features)
        y_test: Labels de test
        model_name: Nom du modèle pour l'affichage (clé unique dans le CSV)
        vectorizer: Vectorizer à appliquer si X_test est du texte
        save: Si True, sauvegarde automatiquement dans le CSV

    Returns:
        dict: Dictionnaire avec les métriques
    """
    if vectorizer:
        X_test_transformed = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_transformed)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*60}")
    print(f" {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"F1-macro:     {f1_macro:.4f}")
    print(f"F1-weighted:  {f1_weighted:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y_test.unique()),
                yticklabels=sorted(y_test.unique()))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    result = {
        'model': model_name,
        'type': 'ML Classique',
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

    if save:
        save_result(result)

    return result


def evaluate_dl_model(model, test_loader, label_encoder, model_name, device=None, save=True):
    """
    Évalue un modèle PyTorch et affiche les métriques.

    Args:
        model: Le modèle PyTorch entraîné
        test_loader: DataLoader de test
        label_encoder: LabelEncoder pour les noms de classes
        model_name: Nom du modèle pour l'affichage (clé unique dans le CSV)
        device: torch device (auto-détecté si None)
        save: Si True, sauvegarde automatiquement dans le CSV

    Returns:
        dict: Dictionnaire avec les métriques
    """
    import torch

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_mac = f1_score(all_labels, all_preds, average='macro')
    f1_w = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n{'='*60}")
    print(f" {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"F1-macro:     {f1_mac:.4f}")
    print(f"F1-weighted:  {f1_w:.4f}")

    target_names = label_encoder.classes_.astype(str)
    print(f"\n{classification_report(all_labels, all_preds, target_names=target_names)}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    result = {
        'model': model_name,
        'type': 'Deep Learning',
        'accuracy': acc,
        'f1_macro': f1_mac,
        'f1_weighted': f1_w
    }

    if save:
        save_result(result)

    return result


def load_results(csv_path=RESULTS_CSV):
    """Charge tous les résultats depuis le CSV."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=['model', 'type', 'accuracy', 'f1_macro', 'f1_weighted'])
