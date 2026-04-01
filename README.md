# 🚁 AeroMind — Détection d'Obstacles Lidar
### Airbus AI Hackathon 2026

Détection et classification d'obstacles 3D (antennes, câbles, pylônes électriques, éoliennes)
dans des nuages de points LiDAR pour hélicoptères en vol basse altitude.

---

## 📁 Structure des fichiers

```
├── inference.py              # ⭐ Script principal d'inférence
├── classifier.pkl            # ⭐ Modèle Random Forest pré-entraîné
├── lidar_utils.py            # Fourni par Airbus — chargement H5, conversion XYZ
├── compute_boxes.py          # DBSCAN + calcul bbox orientée + extraction background
├── generate_final_csv.py     # [TRAIN] Génère labels_train.csv depuis les H5
├── extract_features.py       # [TRAIN] Calcule 22 features géométriques par bbox
├── train_classifier.py       # [TRAIN] Entraîne le Random Forest
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

---

## 🏗️ Architecture du pipeline

```
PHASE 1 — GÉNÉRATION DU GROUND TRUTH (entraînement uniquement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scene_*.h5
  → compute_boxes.py      : DBSCAN par classe RGB + clusters background
  → generate_final_csv.py : aggrège toutes les scènes
  ↓
labels_train_clean.csv  (7 285 bboxes, 5 classes)

PHASE 2 — ENTRAÎNEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
labels_train_clean.csv
  → extract_features.py  : 22 features géométriques par bbox
  ↓
features_train.csv
  → train_classifier.py  : Random Forest 200 arbres, class_weight=balanced
  ↓
classifier.pkl

PHASE 3 — INFÉRENCE (Jour J)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
eval_scene*.h5
  → Détection verticale : grille XY 2m, z_range > 25m, empreinte XY < 5m
  → HDBSCAN             : epsilon=10m, sur points > seuil sol
  → Filtrage géométrique : rejette végétation et terrain
  → Random Forest       : 22 features → 4 classes d'obstacles
  → Post-filtrage + fusion des doublons
  ↓
predictions_scene*.csv
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Inférence (Jour J)


```bash
python inference.py --file eval_sceneA_100.h5 --output predictions_sceneA_100.csv
```

### Générer les 8 CSV de soumission
```bash
mkdir -p predictions/
for f in eval_sceneA_100.h5 eval_sceneA_75.h5 eval_sceneA_50.h5 eval_sceneA_25.h5 \
          eval_sceneB_100.h5 eval_sceneB_75.h5 eval_sceneB_50.h5 eval_sceneB_25.h5; do
    python inference.py --file $f --output predictions/${f%.h5}_predictions.csv
done
```

---

## 🔁 Ré-entraînement (optionnel)

```bash
# Étape 1 — Générer le dataset 
python generate_final_csv.py
# Output : labels_train_clean.csv

# Étape 2 — Extraire les features
python extract_features.py --csv labels_train_clean.csv --output features_train.csv

# Étape 3 — Entraîner le modèle
python train_classifier.py --features features_train.csv --output classifier.pkl --n-trees 200
```

---

## 📊 Performances du modèle

### Classifieur (Random Forest)

| Métrique | Score |
|---|---|
| Accuracy test set | 96.3% |
| F1-macro (cross-val 5-fold) | 0.900 |
| Nombre de paramètres | ~225 500 |
| Format | `.pkl` (scikit-learn) |
| GPU requis | Non — CPU uniquement |

### Dataset d'entraînement

| Classe | ID | Exemples | % |
|---|---|---|---|
| Background | 4 | 5 346 | 73.4% |
| Wind turbine | 3 | 588 | 8.1% |
| Antenna | 0 | 569 | 7.8% |
| Cable | 1 | 557 | 7.6% |
| Electric pole | 2 | 224 | 3.1% |
| **Total** | | **7 284** | |

---

### Légende des couleurs
| Couleur | Signification |
|---|---|
| Blanc | Ground Truth |
| Bleu `(38, 23, 180)` | Antenna |
| Orange `(177, 132, 47)` | Cable |
| Violet `(129, 81, 97)` | Electric pole |
| Vert `(66, 132, 9)` | Wind turbine |

---

## 📋 Format CSV de sortie

| Colonne | Description |
|---|---|
| `ego_x/y/z/yaw` | Pose du véhicule (copiée depuis le H5) |
| `bbox_center_x/y/z` | Centre de la bbox en mètres (repère Lidar local) |
| `bbox_width/length/height` | Dimensions en mètres |
| `bbox_yaw` | Rotation autour de Z |
| `class_ID` | 0=Antenna, 1=Cable, 2=Electric Pole, 3=Wind Turbine |
| `class_label` | Nom textuel de la classe |


## 🧪 Expérimentation Alternative : Détection 3D (PointPillars)

En complément du classifieur Random Forest, nous avons benchmarqué une approche par **Deep Learning** État de l'Art pour évaluer la performance de détection 3D directe sur les nuages de points LiDAR.

### ⚙️ Configuration Technique

* **Framework** : OpenPCDet (Open-MMLab), une boîte à outils professionnelle pour la détection 3D.
* **Modèle** : **PointPillars**, une architecture optimisée pour l'inférence en temps réel sur des nuages de points voxélisés.
* **Classes Cibles** : Configuration spécifique pour les 4 classes d'obstacles Airbus (Antennes, Câbles, Pylônes électriques, Éoliennes).
* **Accélération Matérielle** : Compilation JIT (Just-In-Time) de **780 kernels CUDA** spécialisés pour les GPU NVIDIA T4/P100.

### 📂 Ingénierie des Données

* **Conversion de Format** : Transformation des données Airbus H5 au format **KITTI tracking** pour exploiter les pipelines de détection 3D standardisés.
* **Génération de Métadonnées** : Création de fichiers d'indexation (`.pkl`) pour **912 échantillons d'entraînement**, réduisant drastiquement les temps de lecture sur GPU.
* **Gestion des Contraintes** : Développement d'un script de génération de données "dummy" (images et calibrations) pour satisfaire les exigences du framework tout en conservant un flux d'entraînement 100% LiDAR.

### 🛠️ Défis Techniques Surmontés

* **Optimisation de l'Environnement** : Résolution des conflits critiques de compilation C++ entre `spconv`, `ninja` et `tensorview` sous Python 3.12.
* **Gestion des Ressources** : Migration réussie de la charge de travail vers Kaggle pour contourner les quotas GPU et garantir la stabilité lors des phases de compilation intensive.

