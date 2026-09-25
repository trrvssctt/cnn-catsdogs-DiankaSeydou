# CNN Cats vs Dogs — From Scratch vs Transfer Learning

## Objectif du projet

Ce projet compare deux approches pour la classification binaire chats vs chiens sur le même jeu de données et avec le même protocole d'évaluation :

- Expérience A : réseau CNN simple construit from scratch
- Expérience B : ResNet-18 pré-entraîné sur ImageNet avec fine-tuning de la tête de classification

L'objectif est de mesurer l'impact du transfert learning sur la convergence, la précision et la robustesse du modèle.

---

## Environnement

### Option 1 : pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2 : conda

```bash
conda env create -f environment.yml
conda activate catsdogs
```

Le notebook a été développé avec PyTorch, TorchVision, scikit-learn, NumPy et Matplotlib. Si un GPU est disponible, le code l'utilise automatiquement (`cuda`), ce qui accélère fortement l'entraînement.

---

## Organisation des données

Télécharger le jeu de données Cats vs Dogs et le placer avec la structure suivante :

```text
Cat_Dog_data/
├── train/
│   ├── cat/
│   │   ├── cat_0001.jpg
│   │   └── ...
│   └── dog/
│       ├── dog_0001.jpg
│       └── ...
└── test/
    ├── cat/
    │   ├── cat_0001.jpg
    │   └── ...
    └── dog/
        ├── dog_0001.jpg
        └── ...
```

Le dossier `train/` est ensuite divisé en train/validation avec un split stratifié ; le dossier `test/` reste réservé à l'évaluation finale, comme dans le notebook.

---

## Entraînement

Les hyperparamètres du projet sont inspirés de la configuration du notebook :

- taille d'image : 224x224
- batch size : 64
- optimiser : Adam ou SGD
- poids : 1e-4
- scheduler : cosine annealing
- dropout : 0.5 pour le CNN from scratch, 0.3 pour le transfert learning
- epochs : 15 (scratch), 6 (transfert learning)

### 1) CNN from scratch

```bash
python train.py \
  --model scratch \
  --epochs 15 \
  --batch-size 64 \
  --lr 1e-3 \
  --optimizer adam \
  --dropout 0.5 \
  --scheduler cosine \
  --weight-decay 1e-4
```

Ce modèle comporte 4 blocs convolutionnels avec BatchNorm et Dropout, puis une tête de classification.

### 2) Transfert learning (ResNet-18)

```bash
python train.py \
  --model resnet18 \
  --pretrained \
  --epochs 6 \
  --batch-size 64 \
  --lr 1e-4 \
  --optimizer adam \
  --dropout 0.3 \
  --scheduler cosine \
  --backbone-lr-mult 0.1 \
  --weight-decay 1e-4
```

Pour un réseau de transfert learning avec backbone gelé :

```bash
python train.py \
  --model resnet18 \
  --pretrained \
  --freeze-backbone \
  --epochs 6 \
  --batch-size 64 \
  --lr 1e-4 \
  --optimizer adam \
  --dropout 0.3 \
  --scheduler cosine
```

Le modèle pré-entraîné sur ImageNet est particulièrement adapté ici, car le domaine est proche et les poids utiles pour les caractéristiques visuelles sont déjà appris.

---

## Évaluation et rechargement du meilleur checkpoint

Les meilleurs modèles sont enregistrés dans le dossier `checkpoints/` sous forme de fichiers `.pt` ou `.pth`.

### Évaluation d'un checkpoint

```bash
python evaluate.py \
  --checkpoint checkpoints/best_scratch.pt \
  --data-dir Cat_Dog_data/test \
  --model scratch
```

```bash
python evaluate.py \
  --checkpoint checkpoints/best_resnet18.pt \
  --data-dir Cat_Dog_data/test \
  --model resnet18 \
  --pretrained
```

Le script d'évaluation renvoie la loss, l'accuracy, la précision, le recall, le F1 et la matrice de confusion sur l'ensemble de test.

---

## Résultats

Les expérimentations du notebook montrent un écart très net entre le CNN from scratch et le transfert learning.

| Modèle | Meilleur val acc | Test acc | Précision | Recall | F1 | Observation |
|---|---:|---:|---:|---:|---:|---|
| CNN from scratch | 0.8631 | 0.8476 | ~0.85 | ~0.83 | ~0.84 | Convergence plus lente, surapprentissage plus rapide |
| ResNet-18 fine-tuned | 0.9891 | 0.9872 | ~0.99 | ~0.98 | ~0.98 | Très forte généralisation |
| ResNet-18 backbone gelé | 0.9856 | 0.9848 | ~0.98 | ~0.98 | ~0.98 | Très bon résultat, légèrement inférieur au fine-tuning |

### Courbes observées

- loss train / loss val : décroissance rapide pour ResNet-18, plus progressive pour le CNN scratch
- accuracy train / val : le transfert learning atteint très rapidement une précision élevée, puis se stabilise
- précision / recall : meilleurs résultats pour le modèle pré-entraîné, avec moins de faux positifs et de faux négatifs

### Analyse

Le CNN from scratch atteint des performances raisonnables, mais il est nettement limité par la taille du jeu de données et par le fait qu'il apprend toutes les caractéristiques visuelles depuis zéro. Sans pré-entraînement, le réseau doit apprendre les motifs de base de la détection d'objets à partir d'un volume de données plus faible, ce qui ralentit la convergence et augmente le risque de sous-apprentissage ou de surapprentissage.

Le transfert learning, et en particulier le fine-tuning du ResNet-18, produit une amélioration majeure. Les poids ImageNet apportent des filtres déjà adaptés à la détection des contours, textures et formes, ce qui permet au modèle de converger beaucoup plus vite et de mieux généraliser sur les images de chats et de chiens. Les scores sur validation et test restent très proches, ce qui suggère une bonne stabilité et un faible overfitting.

Le modèle avec backbone gelé donne aussi de très bons résultats, ce qui confirme que les caractéristiques générées par le réseau pré-entraîné sont déjà pertinentes. En revanche, le fine-tuning complet reste légèrement meilleur, car il autorise la remise à jour des filtres du backbone pour mieux s'adapter au problème spécifique de classification chats/chats vs chiens.

---

## Limites et pistes d'amélioration

- Le jeu est binaire et relativement simple ; il serait intéressant de tester d'autres jeux de données plus difficiles.
- L'augmentation peut être enrichie avec bruit, flou, variation de luminosité et recadrage plus agressif.
- Ajouter une validation croisée ou un ensemble de seeds permettrait de mesurer la stabilité des performances.
- Tester d'autres backbones (MobileNetV2, EfficientNet, DenseNet) pourrait apporter une meilleure comparaison entre architectures.
- L'analyse visuelle des erreurs via matrices de confusion et cartes d'attention (Grad-CAM) permettrait d'identifier les cas difficiles.

---

## Structure du dépôt

```text
cnn-catsdogs-NomPrenom/
├── notebook.ipynb
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── checkpoints/
├── runs/
├── results/
├── figures/
└── data/
```

Les dossiers `checkpoints/`, `runs/`, `results/`, `figures/` et les fichiers de poids (`*.pt`, `*.pth`) ne doivent pas être versionnés. Ils sont ignorés via `.gitignore`.
