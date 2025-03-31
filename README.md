# Analyse des couleurs d’une image et impact du daltonisme (CUDA)

## Objectif

L'objectif de ce projet est d'accélérer un code séquentiel utilisant CUDA pour analyser les couleurs d'une image et étudier l'impact du daltonisme. Ce projet repose sur la conversion du traitement d'image séquentiel en un modèle parallèle afin d'améliorer les performances d'analyse des couleurs, en particulier pour les individus daltoniens.

## Méthode

### Étapes principales :
1. **Chargement de l'image** : L'image est chargée dans la mémoire GPU pour permettre une analyse rapide.
2. **Boucle pixel-par-pixel** : Chaque pixel est analysé pour déterminer la couleur dominante (rouge, vert, bleu).
3. **Identification des pixels problématiques pour les daltoniens** : Une fois les pixels analysés, ceux qui sont problématiques pour les daltoniens sont identifiés et marqués.
4. **Utilisation des Kernels CUDA** : 
   - Un kernel CUDA est créé pour analyser chaque pixel en parallèle, avec un thread par pixel.
   - La division du travail se fait en blocs de 16×16 pixels (256 threads) pour tirer parti des multiprocesseurs du GPU.

### Optimisation des accès mémoire
- Les **accès mémoire globaux** sont parallélisés grâce à CUDA, permettant une lecture simultanée et rapide des données.
- Les **accès mémoire coalescés** assurent une efficacité maximale lors des transferts de données entre la mémoire principale et le GPU.

### Synchronisation et comptage atomique
- **Atomic operations** : La fonction `cuda.atomic.add()` est utilisée pour garantir le comptage des pixels sans conflits, assurant ainsi la cohérence des données en mémoire globale sans avoir besoin de verrous.

## Performances

### Comparaison CPU vs GPU (CUDA)
| Critère                | CPU           | GPU (CUDA)   | Gain             |
|------------------------|---------------|--------------|------------------|
| Temps d'exécution (5 images) | 1.46 s        | 0.13 s       | ~11x plus rapide |
| Scalabilité            | Mauvaise      | Excellente   |                  |
| Parallélisme GPU       | Inefficace    | Efficace     |                  |
| Accès mémoire          | Séquentiel    | Optimisé     |                  |
| Sécurisation des accès | Aucune        | `cuda.atomic.add()` |                  |

### Détails des optimisations
- **Utilisation du GPU** : L'exploitation des multiprocesseurs du GPU permet d’accélérer le calcul.
- **Accès à la mémoire globale** : Parallélisation des accès mémoire, avec des optimisations pour réduire les conflits mémoire (bank conflicts).
- **Mémoire partagée** : Utilisation optimisée de la mémoire partagée pour réduire les accès à la mémoire globale, augmentant ainsi l'efficacité.

### Impact sur le daltonisme
- Le code propose un support amélioré pour tous les types de daltonisme, en utilisant des filtres plus précis.
- Passage d’une simple détection de la couleur verte à une analyse fine des confusions spectrales (notamment la **deutéranopie**).
- Détection de **zones localisées** avec une forte concentration de couleurs non discernables par les daltoniens, plutôt que de se baser sur un pourcentage global.

### Performances sur images
- **Accélération** : Le traitement des images est jusqu’à **11 fois plus rapide** en utilisant CUDA (sur 5 images). Les gains peuvent être encore plus importants sur des volumes de données plus larges.

### Limitations
- **Processus lent** : Le processus séquentiel initial présente des accès mémoire coûteux, notamment lors de la lecture/écriture en mémoire principale sans optimisation.
- **Accès mémoire coûteux** : Le processus séquentiel souffre de lectures/écritures en mémoire principale sans optimisation, ce qui ralentit les performances.

## Perspectives

- **Support multi-GPU** : Utilisation de plusieurs GPU pour le traitement massif d’images.
- **Pipeline complet d’amélioration d’image** : Détection des zones problématiques, application du filtre adapté, et vérification post-correction.
- **Analyse fine des confusions spectrales** : Amélioration des algorithmes pour détecter des confusions spectrales spécifiques, notamment pour les personnes atteintes de différentes formes de daltonisme.

## Installation

1. **Prérequis** :
   - NVIDIA GPU avec support CUDA.
   - Driver CUDA compatible.
   - Python et les librairies nécessaires (NumPy, OpenCV, etc.).
   - CUDA Toolkit installé.

2. **Installation des dépendances** :
   ```bash
   pip install numpy opencv-python pycuda
   ```

3. **Compilation du code CUDA** : Utiliser `nvcc` pour compiler les kernels CUDA et l'intégrer au code Python.

## Exécution

Pour exécuter le programme, il suffit de lancer le script Python en fournissant l'image à analyser. Le programme traitera l'image et affichera les résultats, y compris les zones identifiées comme problématiques pour les daltoniens.
