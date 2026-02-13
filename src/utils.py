import json
import re
import string
import pandas as pd

def load_json_lines(file_path:str, n_lines:int=None) -> pd.DataFrame:
    """
    Charge un fichier JSONL ligne par ligne et retourne un DataFrame.
    Args:
        file_path (str): Le chemin vers le fichier JSONL.
        n_lines (int, optional): Nombre de lignes à charger (pour test). Par défaut, None pour charger tout.
    Returns:
        pd.DataFrame: Un DataFrame contenant les données chargées.
    """

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_lines and i >= n_lines:
                break
            data.append(json.loads(line))
            if (i + 1) % 100000 == 0:
                print(f"  Chargé {i + 1:,} lignes...")
    return pd.DataFrame(data)

def clean_text(text: str, lowercase: bool = True, remove_punctuation: bool = False, remove_numbers: bool = False) -> str:
    """
    Nettoie le texte en appliquant plusieurs étapes de prétraitement.
    Args:
        text (str): Le texte à nettoyer.
        lowercase (bool): Si True, met le texte en minuscules.
        remove_punctuation (bool): Si True, enlève la ponctuation du texte.
        remove_numbers (bool): Si True, enlève les chiffres du texte.
    Returns:
        str: Le texte nettoyé.
    """
    
    # Lowercase
    if lowercase:
        text = text.lower()

    # Enlever URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Enlever mentions/hashtags (si pertinent)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Enlever la ponctuation (optionnel, dépend du modèle)
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Enlever les chiffres (optionnel)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Enlever espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def truncate_review(text: str, max_words: int = 200) -> str:
    """
    Truncate le texte à un nombre maximum de mots.
    Args:
        text (str): Le texte à tronquer.
        max_words (int): Le nombre maximum de mots à conserver.
    Returns:
        str: Le texte tronqué.
    """
    words = text.split()
    return ' '.join(words[:max_words])

def create_polarity_label(rating: float) -> str:
    """
    Convertit une note (1-5 étoiles) en label de polarité.
    
    Args:
        rating (float): La note de 1 à 5 étoiles.
    
    Returns:
        str: 'positive', 'negative' ou 'neutral'.
    """
    if rating > 3:
        return 'positive'
    elif rating < 3:
        return 'negative'
    else:
        return 'neutral'


def add_polarity_column(df, rating_column: str = 'stars', polarity_column: str = 'polarite'):
    """
    Ajoute une colonne de polarité au DataFrame.
    
    Args:
        df: Le DataFrame à modifier.
        rating_column (str): Le nom de la colonne contenant les notes.
        polarity_column (str): Le nom de la nouvelle colonne de polarité.
    
    Returns:
        DataFrame: Le DataFrame avec la nouvelle colonne ajoutée.
    """
    df[polarity_column] = df[rating_column].apply(create_polarity_label)
    return df

def add_rating_column(df, rating_column: str = 'stars', target_column: str = 'rating'):
    """
    Copie la colonne de notes vers une colonne de target pour la modélisation.
    
    Args:
        df: Le DataFrame à modifier.
        rating_column (str): Le nom de la colonne contenant les notes.
        target_column (str): Le nom de la nouvelle colonne target.
    
    Returns:
        DataFrame: Le DataFrame avec la nouvelle colonne ajoutée.
    """
    df[target_column] = df[rating_column]
    return df


def remove_short_reviews(df, text_column: str = 'text', min_words: int = 10):
    """
    Supprime les reviews trop courtes.
    
    Args:
        df: Le DataFrame à filtrer.
        text_column (str): Le nom de la colonne contenant le texte.
        min_words (int): Nombre minimum de mots requis.
    
    Returns:
        DataFrame: Le DataFrame filtré.
    """
    return df[df[text_column].str.split().str.len() >= min_words].reset_index(drop=True)


def preprocess_dataframe(df: pd.DataFrame, 
                         text_column: str = 'text',
                         rating_column: str = 'stars',
                         lowercase: bool = True,
                         remove_punctuation: bool = False,
                         remove_numbers: bool = False,
                         add_polarity: bool = True,
                         add_rating: bool = True,
                         min_words: int = 10,
                         truncate: bool = False,
                         max_words: int = 200):
    """
    Pipeline complet de preprocessing pour le DataFrame.
    
    Args:
        df: Le DataFrame à prétraiter.
        text_column (str): Nom de la colonne texte.
        rating_column (str): Nom de la colonne rating.
        lowercase (bool): Mettre le texte en minuscules.
        remove_punctuation (bool): Enlever la ponctuation.
        remove_numbers (bool): Enlever les chiffres.
        add_polarity (bool): Ajouter la colonne polarité.
        add_rating (bool): Ajouter la colonne rating.
        min_words (int): Nombre minimum de mots par review.
        truncate (bool): Tronquer les reviews trop longues.
        max_words (int): Nombre maximum de mots si truncate=True.
    
    Returns:
        DataFrame: Le DataFrame prétraité.
    """
    # Copie pour ne pas modifier l'original
    df = df.copy()
    
    # Supprimer les NaN
    df = df.dropna(subset=[text_column, rating_column])
    
    # Nettoyer le texte
    df['text_clean'] = df[text_column].apply(lambda x: clean_text(x, lowercase=lowercase, remove_punctuation=remove_punctuation, remove_numbers=remove_numbers))

    # Tronquer si demandé
    if truncate:
        df['text_clean'] = df['text_clean'].apply(lambda x: truncate_review(x, max_words))
    
    # Supprimer les reviews trop courtes
    df = remove_short_reviews(df, text_column='text_clean', min_words=min_words)
    
    # Ajouter les labels
    if add_polarity:
        df = add_polarity_column(df, rating_column=rating_column)
    
    if add_rating:
        df = add_rating_column(df, rating_column=rating_column)
    
    return df
