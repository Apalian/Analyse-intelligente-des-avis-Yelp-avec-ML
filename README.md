# Analyse intelligente des avis Yelp avec ML, Deep Learning et IA agentique

Projet SAE S6.C.01 - Machine Learning & Deep Learning

## Description du projet

Ce projet vise à analyser les avis Yelp en utilisant différentes techniques de Machine Learning, Deep Learning et IA générative. Les objectifs principaux sont:

1. **Analyse exploratoire** des données Yelp
2. **Prédiction de la polarité** des commentaires (positif/négatif/neutre)
3. **Prédiction du score** (1 à 5 étoiles)
4. **Extraction d'aspects** avec analyse de sentiment (ABSA)
5. **Classification zero-shot/few-shot** avec LLM

## Structure du projet

```
Analyse-intelligente-des-avis-Yelp-avec-ML/
├── data/                          # Données Yelp (non versionnées)
│   ├── yelp_academic_dataset_review.json
│   ├── yelp_academic_dataset_business.json
│   └── yelp_academic_dataset_user.json
├── notebooks/
│   └── phase_A/                   # Analyse exploratoire
│       ├── 00_data_loading.ipynb
│       ├── 01_distribution_analysis.ipynb
│       ├── 02_reviewer_analysis.ipynb
│       └── 03_text_analysis.ipynb
├── src/
│   └── utils/                     # Fonctions utilitaires
├── figures/                       # Graphiques générés
├── requirements.txt               # Dépendances Python
├── .gitignore
└── README.md
```

## Installation

### 1. Cloner le repository

```bash
git clone git@github.com:Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML.git
cd Analyse-intelligente-des-avis-Yelp-avec-ML
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset Yelp

1. Aller sur [Yelp Open Dataset](https://www.yelp.com/dataset)
2. Télécharger le dataset
3. Extraire les fichiers JSON dans le dossier `data/`

## Phase A - Analyse Exploratoire

Les notebooks de la Phase A effectuent des analyses statistiques et visuelles sur le dataset Yelp.

### Notebooks disponibles:

#### [00_data_loading.ipynb](notebooks/phase_A/00_data_loading.ipynb)
- Chargement des fichiers JSON Lines
- Exploration initiale des données
- Préparation et nettoyage
- Création de colonnes dérivées (longueur, polarité)

#### [01_distribution_analysis.ipynb](notebooks/phase_A/01_distribution_analysis.ipynb)
**Analyses:**
- Répartition des avis par catégorie de business
- Lien entre popularité et note moyenne
- Les businesses populaires sont-ils jugés plus sévèrement?

#### [02_reviewer_analysis.ipynb](notebooks/phase_A/02_reviewer_analysis.ipynb)
**Analyses:**
- Les "gros reviewers" sont-ils plus sévères?
- Les utilisateurs expérimentés font-ils des reviews plus détaillées?
- Analyse de la variabilité des notes

#### [03_text_analysis.ipynb](notebooks/phase_A/03_text_analysis.ipynb)
**Analyses:**
- Longueur des reviews par classe de note (1-5 étoiles)
- Les avis négatifs sont-ils plus longs que les positifs?
- Comparaison des vocabulaires (TF-IDF)
- Top 10 mots caractéristiques négatifs vs positifs
- Word clouds (optionnel)

### Ordre d'exécution recommandé:

1. Commencer par `00_data_loading.ipynb` pour charger les données
2. Exécuter les notebooks 01-03 dans l'ordre (ou selon votre intérêt)
3. Chaque notebook est autonome et peut être exécuté indépendamment

### Tips:

- **Échantillonnage**: Pour tester rapidement, utilisez le paramètre `SAMPLE_SIZE` dans les notebooks (ex: 100000 reviews au lieu de millions)
- **Mémoire**: Si vous manquez de RAM, réduisez la taille de l'échantillon
- **Sauvegarde**: Les notebooks sauvent automatiquement les résultats dans `data/`
- **Figures**: Toutes les visualisations sont sauvegardées dans `figures/`

## Phase B - Modèles de Prédiction

### Tâches à réaliser:

1. **Prédiction de la polarité** (positif/négatif/neutre)
   - Règle: score > 3 → positif, score < 3 → négatif, score = 3 → neutre

2. **Prédiction du score** (1 à 5 étoiles)

### Représentations du texte à tester:
- Bag-of-Words
- TF-IDF
- Embeddings (BERT, GPT)

### Modèles à implémenter:
- Algorithmes classiques (Régression logistique, SVM)
- Deep Learning (MLP, CNN)
- Transformers (fine-tuning)

## Phase C - IA Générative

### 1. Classification zero-shot et few-shot
- Utiliser un LLM pour prédire la polarité sans entraînement
- Tester avec quelques exemples (few-shot)

### 2. Extraction d'aspects (ABSA)
- Identifier les aspects mentionnés (nourriture, service, prix, etc.)
- Déterminer le sentiment pour chaque aspect
- Utiliser LangChain/LlamaIndex

## Issues GitHub

Le projet est organisé en issues GitHub pour suivre l'avancement:

- [Issue #1](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/1): Documentation et rapport final
- [Issue #2](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/2): Setup et préparation
- [Issue #3](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/3): Phase A - Analyse exploratoire
- [Issue #4](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/4): Phase B.1 - Prédiction polarité
- [Issue #5](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/5): Phase B.2 - Prédiction score
- [Issue #6](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/6): Phase C.1 - Zero-shot/few-shot
- [Issue #7](https://github.com/Apalian/Analyse-intelligente-des-avis-Yelp-avec-ML/issues/7): Phase C.2 - Extraction d'aspects

## Ressources

### Dataset
- [Yelp Open Dataset](https://www.yelp.com/dataset)
- [Documentation du dataset](https://www.yelp.com/dataset/documentation/main)

### Références
- [Scikit-learn documentation](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://docs.llamaindex.ai/)

## Auteurs

- Projet SAE S6.C.01
- IUT Toulouse III - Paul Sabatier
- Janvier 2026

## Licence

Ce projet est réalisé dans un cadre pédagogique.
