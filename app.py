from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import threading
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
THUMBNAILS_FOLDER = 'thumbnails'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MODEL_PATH = 'my_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['THUMBNAILS_FOLDER'] = THUMBNAILS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Créer les dossiers
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)

# Charger le modèle
try:
    model = load_model(MODEL_PATH)
    print("✓ Modèle chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    model = None

# Classes du modèle
CLASS_NAMES = [
    'Abuse', 'Assault', 'Burglary', 'Explosion', 
    'Fighting', 'NormalVideos', 'Shooting', 
    'Shoplifting', 'Stealing'
]

CLASS_COLORS = {
    'Abuse': (0, 0, 255),
    'Assault': (0, 69, 255),
    'Burglary': (0, 140, 255),
    'Explosion': (0, 0, 139),
    'Fighting': (255, 0, 255),
    'NormalVideos': (0, 255, 0),
    'Shooting': (128, 0, 128),
    'Shoplifting': (255, 255, 0),
    'Stealing': (0, 165, 255)
}

# Variables globales
current_video = None
is_processing = False
webcam_active = False
anomaly_detections = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame):
    """Prétraiter une frame pour le modèle (Grayscale - 1 channel comme dans l'entraînement)"""
    img = cv2.resize(frame, (48, 48))
    # Convertir en niveaux de gris comme dans l'entraînement
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    # Ajouter la dimension du canal (48, 48, 1)
    img = np.expand_dims(img, axis=-1)
    # Ajouter la dimension du batch (1, 48, 48, 1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_frame(frame):
    """Prédire la classe d'une frame"""
    if model is None:
        return "Model Error", 0.0, False, None
    
    preprocessed = preprocess_frame(frame)
    prediction = model.predict(preprocessed, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx])
    predicted_class = CLASS_NAMES[class_idx]
    
    # Anomalie si ce n'est pas "NormalVideos" et confiance > 60%
    is_anomaly = predicted_class != 'NormalVideos' and confidence > 0.6
    
    # Retourner aussi toutes les probabilités pour analyse
    all_probs = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, is_anomaly, all_probs

def detect_anomaly_regions(frame, predicted_class, confidence):
    """Détection de régions d'anomalies (simulation - peut être amélioré avec object detection)"""
    h, w = frame.shape[:2]
    
    # Pour une vraie détection, utiliser un modèle de détection d'objets
    # Ici, on crée des régions basées sur l'intensité des pixels
    regions = []
    
    if confidence > 0.6:
        # Convertir en HSV pour détecter des zones d'intérêt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Détecter les zones avec mouvement/changement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer et créer des boîtes englobantes
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Seuil minimum
                x, y, w_box, h_box = cv2.boundingRect(contour)
                regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w_box),
                    'height': int(h_box),
                    'confidence': confidence
                })
    
    # Si aucune région détectée mais anomalie, créer une région centrale
    if len(regions) == 0 and confidence > 0.6:
        regions.append({
            'x': int(w * 0.25),
            'y': int(h * 0.25),
            'width': int(w * 0.5),
            'height': int(h * 0.5),
            'confidence': confidence
        })
    
    return regions

def draw_detection_overlay(frame, predicted_class, confidence, is_anomaly, frame_number, thermal_mode=False, show_regions=True):
    """Dessine l'overlay de détection avec encadrement des régions"""
    h, w = frame.shape[:2]
    
    # Détecter les régions d'anomalies
    regions = []
    if is_anomaly and show_regions:
        regions = detect_anomaly_regions(frame, predicted_class, confidence)
        
        # Dessiner les boîtes de détection
        for region in regions:
            x, y, w_box, h_box = region['x'], region['y'], region['width'], region['height']
            color = CLASS_COLORS.get(predicted_class, (0, 0, 255))
            
            # Rectangle principal
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 3)
            
            # Label avec fond
            label = f"{predicted_class} {confidence*100:.0f}%"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Fond du label
            cv2.rectangle(frame, 
                         (x, y - label_size[1] - 10),
                         (x + label_size[0] + 10, y),
                         color, -1)
            
            # Texte du label
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Effet de clignotement pour anomalies critiques
            if confidence > 0.8 and frame_number % 10 < 5:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), color, -1)
                frame = cv2.addWeighted(frame, 0.95, overlay, 0.05, 0)
    
    # Appliquer l'effet thermique si activé
    if thermal_mode and is_anomaly:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.6, thermal, 0.4, 0)
    
    # Cadre rouge général pour anomalie
    if is_anomaly:
        color = (0, 0, 255)
        thickness = 4
        cv2.rectangle(frame, (5, 5), (w-5, h-5), color, thickness)
    
    # Panneau d'information semi-transparent
    panel_height = 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Statut
    status_text = "🚨 ANOMALIE DÉTECTÉE" if is_anomaly else "✓ Normal"
    status_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    cv2.putText(frame, status_text, (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    
    # Classe prédite
    class_text = f"Type: {predicted_class}"
    class_color = CLASS_COLORS.get(predicted_class, (255, 255, 255))
    cv2.putText(frame, class_text, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color, 2)
    
    # Confiance
    conf_text = f"Confiance: {confidence*100:.1f}%"
    cv2.putText(frame, conf_text, (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Nombre de régions détectées
    if len(regions) > 0:
        region_text = f"Regions: {len(regions)}"
        cv2.putText(frame, region_text, (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (w - 280, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame, regions

def save_thumbnail(frame, filename):
    """Sauvegarde une miniature de la frame"""
    thumbnail_path = os.path.join(app.config['THUMBNAILS_FOLDER'], filename)
    cv2.imwrite(thumbnail_path, frame)
    return thumbnail_path

def process_video(video_path, thermal_mode=False, frame_skip=5):
    """Traite une vidéo et génère des frames annotées"""
    global is_processing, anomaly_detections
    is_processing = True
    anomaly_detections = []
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Erreur: Impossible d'ouvrir la vidéo {video_path}")
        is_processing = False
        return
    
    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Valeur par défaut
    
    # Variables pour maintenir la dernière prédiction
    last_predicted_class = "NormalVideos"
    last_confidence = 0.0
    last_is_anomaly = False
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Prédiction uniquement sur les frames sélectionnées
            if frame_number % frame_skip == 0:
                try:
                    predicted_class, confidence, is_anomaly, all_probs = predict_frame(frame)
                    
                    # Mettre à jour les dernières valeurs
                    last_predicted_class = predicted_class
                    last_confidence = confidence
                    last_is_anomaly = is_anomaly
                    
                    # Enregistrer les anomalies
                    if is_anomaly:
                        timestamp_sec = frame_number / fps
                        thumbnail_filename = f"thumb_{frame_number}.jpg"
                        
                        # Sauvegarder miniature
                        try:
                            thumbnail_path = save_thumbnail(frame, thumbnail_filename)
                        except Exception as e:
                            print(f"⚠️ Erreur sauvegarde miniature: {e}")
                            thumbnail_filename = None
                        
                        anomaly_detections.append({
                            'frame': frame_number,
                            'time': float(timestamp_sec),
                            'time_str': f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}",
                            'class': predicted_class,
                            'confidence': float(confidence),
                            'thumbnail': thumbnail_filename,
                            'all_probabilities': all_probs
                        })
                except Exception as e:
                    print(f"❌ Erreur de prédiction frame {frame_number}: {e}")
                    # Continuer avec les dernières valeurs connues
            
            # Dessiner l'overlay avec les dernières valeurs
            try:
                annotated_frame, regions = draw_detection_overlay(
                    frame, last_predicted_class, last_confidence, last_is_anomaly, 
                    frame_number, thermal_mode, show_regions=True
                )
            except Exception as e:
                print(f"❌ Erreur draw_overlay frame {frame_number}: {e}")
                annotated_frame = frame
            
            # Encoder en JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"❌ Erreur dans process_video: {e}")
    
    finally:
        cap.release()
        is_processing = False
        
        # Sauvegarder le rapport
        try:
            report = {
                'video': os.path.basename(video_path),
                'total_frames': frame_number,
                'fps': float(fps),
                'duration': float(frame_number / fps),
                'anomalies_count': len(anomaly_detections),
                'anomalies': anomaly_detections,
                'timestamp': datetime.now().isoformat()
            }
            
            report_path = os.path.join(
                app.config['RESULTS_FOLDER'],
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✓ Rapport sauvegardé: {report_path}")
            print(f"✓ Anomalies détectées: {len(anomaly_detections)}")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde rapport: {e}")
def generate_webcam():
    """Génère le flux webcam avec détection"""
    global webcam_active
    
    camera = cv2.VideoCapture(0)
    
    # Vérifier si la caméra s'est ouverte correctement
    if not camera.isOpened():
        print("❌ Erreur: Impossible d'ouvrir la webcam")
        webcam_active = False
        return
    
    frame_number = 0
    predicted_class = "NormalVideos"
    confidence = 0.0
    is_anomaly = False
    
    try:
        while webcam_active:
            success, frame = camera.read()
            if not success:
                print("❌ Erreur: Impossible de lire la frame de la webcam")
                break
            
            frame_number += 1
            
            # Prédiction toutes les 3 frames pour performance
            if frame_number % 3 == 0:
                try:
                    predicted_class, confidence, is_anomaly, all_probs = predict_frame(frame)
                    
                    # Sauvegarder les anomalies détectées en webcam
                    if is_anomaly:
                        global anomaly_detections
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        anomaly_detections.append({
                            'frame': frame_number,
                            'time': timestamp,
                            'time_str': timestamp,
                            'class': predicted_class,
                            'confidence': float(confidence),
                            'source': 'webcam',
                            'thumbnail': None
                        })
                except Exception as e:
                    print(f"❌ Erreur de prédiction: {e}")
                    continue
            
            # Dessiner l'overlay
            annotated_frame, regions = draw_detection_overlay(
                frame, predicted_class, confidence, is_anomaly,
                frame_number, thermal_mode=False, show_regions=True
            )
            
            # Encoder
            ret, buffer = cv2.imencode('.jpg', annotated_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"❌ Erreur dans generate_webcam: {e}")
    
    finally:
        camera.release()
        webcam_active = False
        print("✓ Webcam fermée")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Page du tableau de bord"""
    return render_template('dashboard.html')

@app.route('/history')
def history():
    """Page de l'historique des analyses"""
    return render_template('history.html')

@app.route('/settings')
def settings():
    """Page des paramètres"""
    return render_template('settings.html')

@app.route('/about')
def about():
    """Page à propos"""
    return render_template('about.html')

@app.route('/api/statistics')
def get_statistics():
    """Retourne les statistiques globales"""
    try:
        # Compter les rapports
        reports = []
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            if filename.endswith('.json'):
                filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                with open(filepath, 'r') as f:
                    reports.append(json.load(f))
        
        # Calculer les statistiques
        total_analyses = len(reports)
        total_anomalies = sum(r.get('anomalies_count', 0) for r in reports)
        
        # Compter par type d'anomalie
        anomaly_types = {}
        for report in reports:
            for anomaly in report.get('anomalies', []):
                cls = anomaly.get('class', 'Unknown')
                anomaly_types[cls] = anomaly_types.get(cls, 0) + 1
        
        # Analyses récentes (7 derniers jours)
        from datetime import timedelta
        recent_date = datetime.now() - timedelta(days=7)
        recent_analyses = 0
        for report in reports:
            report_date = datetime.fromisoformat(report.get('timestamp', ''))
            if report_date > recent_date:
                recent_analyses += 1
        
        return jsonify({
            'total_analyses': total_analyses,
            'total_anomalies': total_anomalies,
            'recent_analyses': recent_analyses,
            'anomaly_types': anomaly_types,
            'reports': reports[-10:]  # 10 derniers rapports
        })
    except Exception as e:
        print(f"❌ Erreur statistiques: {e}")
        return jsonify({
            'total_analyses': 0,
            'total_anomalies': 0,
            'recent_analyses': 0,
            'anomaly_types': {},
            'reports': []
        })

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Aucune vidéo fournie'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Vidéo uploadée avec succès'
        })
    
    return jsonify({'error': 'Format de fichier non autorisé'}), 400

@app.route('/video_feed/<filename>')
def video_feed(filename):
    global current_video
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Vidéo non trouvée'}), 404
    
    current_video = video_path
    
    thermal_mode = request.args.get('thermal', 'false').lower() == 'true'
    frame_skip = int(request.args.get('skip', 5))
    
    return Response(
        process_video(video_path, thermal_mode, frame_skip),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/webcam_feed')
def webcam_feed():
    global webcam_active
    webcam_active = True
    
    return Response(
        generate_webcam(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active, anomaly_detections
    webcam_active = True
    anomaly_detections = []
    return jsonify({'success': True, 'message': 'Webcam démarrée'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({'success': True, 'message': 'Webcam arrêtée'})

@app.route('/status')
def status():
    return jsonify({
        'is_processing': is_processing,
        'webcam_active': webcam_active,
        'model_loaded': model is not None
    })

@app.route('/anomalies')
def get_anomalies():
    """Retourne la liste des anomalies détectées"""
    global anomaly_detections
    return jsonify({
        'anomalies': anomaly_detections,
        'count': len(anomaly_detections)
    })

@app.route('/thumbnail/<filename>')
def get_thumbnail(filename):
    """Sert une miniature"""
    from flask import send_from_directory
    return send_from_directory(app.config['THUMBNAILS_FOLDER'], filename)

@app.route('/reports')
def get_reports():
    reports = []
    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if filename.endswith('.json'):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            with open(filepath, 'r') as f:
                reports.append(json.load(f))
    
    reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify(reports)

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Démarrage de l'application de détection d'anomalies")
    print("=" * 50)
    print(f"✓ Modèle: {'Chargé' if model else 'Non chargé'}")
    print(f"✓ Classes: {len(CLASS_NAMES)}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)