# CNN Cat-Dog

Un réseau de neurones convolutif (CNN) construit avec Keras/TensorFlow pour classer des images de chats et de chiens.

## Architecture du modèle

`model.py` définit et entraîne un CNN `Sequential` qui prend en entrée des images RGB de 150 × 150 pixels :

| Étape | Couches | Rôle |
|-------|---------|------|
| Bloc 1 | `Conv2D` 32 filtres (3×3) + `MaxPooling2D` | Détecte les contours et formes simples |
| Bloc 2 | `Conv2D` 64 filtres (3×3) + `MaxPooling2D` | Détecte des motifs plus complexes |
| Bloc 3 | `Conv2D` 128 filtres (3×3) + `MaxPooling2D` | Détecte des structures de haut niveau |
| Classification | `Flatten` → `Dense(512)` → `Dropout(0.5)` → `Dense(2, softmax)` | Décision finale : chat ou chien |

Chaque bloc convolutif applique la fonction d'activation ReLU, puis réduit la taille des cartes de caractéristiques par pooling. Le `Dropout(0.5)` désactive aléatoirement la moitié des neurones lors de l'entraînement afin de limiter le surapprentissage. La couche de sortie à 2 neurones (softmax) donne la probabilité de chaque classe.

Le modèle est compilé avec l'optimiseur **Adam** et la fonction de coût **entropie croisée catégorielle**, avec la précision comme métrique de suivi.

## Préparation des données

Les images sont chargées depuis deux répertoires locaux (`training_data` et `testing_data`) puis normalisées (valeurs de pixels ramenées entre 0 et 1). Sur l'ensemble d'entraînement, une **augmentation de données** est appliquée à la volée pour enrichir artificiellement le jeu de données et améliorer la généralisation :

- rotation jusqu'à 20°
- zoom jusqu'à 15 %
- décalages horizontaux et verticaux jusqu'à 20 %
- cisaillement jusqu'à 15 %
- retournement horizontal

L'ensemble d'entraînement est ensuite divisé avec `validation_split=0.2` : 80 % des images servent à l'entraînement, 20 % à la validation à chaque époque.

L'entraînement dure **10 époques** avec des lots de **32 images**.

## Évaluation et prédiction

À la fin de l'entraînement, le script :

1. **Sauvegarde** le modèle entraîné dans `cat_dog_model.keras` ;
2. **Évalue** le modèle sur le jeu de test (`testing_data`) et affiche la **précision globale** en pourcentage ;
3. **Prédit** la classe d'une image d'exemple du jeu de test et affiche le résultat dans la console, par exemple :
   ```
   🖼️ Prediction for `chat.101.jpg`: 🐱 Cats
   ```

Le jeu de test n'est jamais vu pendant l'entraînement : la précision affichée mesure donc la capacité du modèle à généraliser à de nouvelles images.

## Structure des données attendue

Le script attend deux répertoires locaux, contenant chacun un sous-dossier par classe (`cats/`, `dogs/`) :

```
training_data/
├── cats/
└── dogs/
testing_data/
├── cats/
└── dogs/
```

Ces répertoires (ainsi que les fichiers de modèle résultants) ne sont pas versionnés — voir `.gitignore` — car le jeu de données et les poids entraînés sont trop volumineux pour le dépôt. Modifiez les constantes `TRAIN_DIR` et `TEST_DIR` en haut de `model.py` pour pointer vers l'emplacement local de votre jeu de données.

## Installation et utilisation

```bash
pip install -r requirements.txt
python3 model.py
```

## Auteur

**Diane Gnabehi** — projet 42 Paris

## License

Distribué sous licence MIT — voir [LICENSE](LICENSE).
