# anomaly-detection-IA-app

### demo

<img width="683" height="806" alt="Capture d’écran 2026-01-14 151051" src="https://github.com/user-attachments/assets/f38fee56-5024-4355-ac61-a1c00585636d" />
<img width="1920" height="1027" alt="Capture d’écran 2026-01-14 144908" src="https://github.com/user-attachments/assets/8c4cac48-f4a0-4d1f-9360-7ce3f8aac652" />
<img width="1920" height="1031" alt="Capture d’écran 2026-01-14 145324" src="https://github.com/user-attachments/assets/2a37c830-ad15-477f-a350-d4f6d6c3027d" />
<img width="1917" height="1030" alt="Capture d’écran 2026-01-14 145222" src="https://github.com/user-attachments/assets/a704d06d-cce3-4f27-a856-34a2e61147ac" />
<img width="1897" height="875" alt="Capture d’écran 2026-01-14 145451" src="https://github.com/user-attachments/assets/9797576c-095a-47fc-b1ae-f61810db89ea" />
<img width="1383" height="838" alt="Capture d’écran 2026-01-14 153434" src="https://github.com/user-attachments/assets/7655382f-e51c-485e-ab12-3beed77cdc0b" />
<img width="1427" height="766" alt="Capture d’écran 2026-01-14 154938" src="https://github.com/user-attachments/assets/1a03842a-4dfa-4507-aba0-c21a121d82ce" />
<img width="1387" height="876" alt="Capture d’écran 2026-01-14 151146" src="https://github.com/user-attachments/assets/b79dfc49-673c-4960-af57-183b12e1caed" />
<img width="788" height="847" alt="Capture d’écran 2026-01-14 151431" src="https://github.com/user-attachments/assets/d337ba9a-bab3-448a-a9b0-0c4f3b332a9f" />
<img width="787" height="741" alt="Capture d’écran 2026-01-14 151605" src="https://github.com/user-attachments/assets/2fbccb6f-61cb-4860-9406-f1571ccae019" />
<img width="1032" height="841" alt="Capture d’écran 2026-01-14 151812" src="https://github.com/user-attachments/assets/c4a55b58-4a27-4638-ad5d-bf6830512038" />

# 🎬 Système de Détection d'Anomalies Vidéo

Application web Flask utilisant un modèle de Deep Learning (Autoencoder 3D CNN) pour détecter automatiquement des anomalies dans des vidéos de surveillance.


## 🎯 Aperçu

Ce projet détecte automatiquement des comportements anormaux dans des vidéos de surveillance (zones publiques, parkings, etc.) en utilisant:
- **Modèle**: Autoencoder 3D Convolutionnel
- **Datasets d'entraînement**: UCSD Pedestrian (Ped1 & Ped2) + CUHK Avenue
- **Framework**: TensorFlow/Keras + Flask
- **Interface**: Web responsive avec drag & drop

### Fonctionnalités
✅ Upload de vidéos (MP4, AVI, MOV, MKV) 
✅ Détection webcam en temps réel
✅ Détection automatique d'anomalies frame par frame  
✅ Identification de segments suspects avec timestamps  
✅ Visualisation des frames avec overlays  
✅ Statistiques détaillées (scores, pourcentages, etc.)  
✅ Interface web moderne et intuitive  

### Stack Technique
- **Backend**: Flask (Python)
- **ML/DL**: TensorFlow 2.15, Keras
- **Vision**: OpenCV
- **Frontend**: flask app

### anomaly_detection_training.ipynb
- **Entraînement**: 60 epochs avec Early Stopping
- **Validation Split**: 15%
- **Threshold global**: 0.005247
- **Architecture**: 3D CNN Autoencoder (32→64→128 filters)
- Détection d'anomalies majeures: ✅ Bon
- Faux positifs: Modéré
- Temps d'inférence: ~2-5 min pour 200 frames


