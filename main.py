import serial
import time
import cv2
import mediapipe as mp
import threading
import numpy as np
from flask import Flask, Response, jsonify, render_template_string
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Serial configuration
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

app = Flask(__name__)
arduino = None
lock = threading.Lock()

# Current finger angles (0-180)
finger_angles = {
    'thumb': 180,
    'index': 180,
    'middle': 180,
    'ring': 180,
    'pinky': 180
}

MODEL_PATH = "hand_landmarker.task"

def download_model():
    import urllib.request
    import os
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded!")

camera = cv2.VideoCapture(0)

def connect_arduino():
    global arduino
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {SERIAL_PORT}")
        return True
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return False

def send_finger_angle(finger_idx, angle):
    """Send single finger angle: F:finger,angle"""
    if arduino and arduino.is_open:
        try:
            cmd = f"F:{finger_idx},{angle}\n"
            arduino.write(cmd.encode())
            return True
        except:
            return False
    return False

def send_all_angles(angles):
    """Send all finger angles at once: A:t,i,m,r,p"""
    if arduino and arduino.is_open:
        try:
            cmd = f"A:{angles[0]},{angles[1]},{angles[2]},{angles[3]},{angles[4]}\n"
            arduino.write(cmd.encode())
            return True
        except:
            return False
    return False

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in degrees"""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_finger_curl(landmarks, finger_indices):
    """
    Calculate finger curl based on joint angles.
    Returns 0-180 where 0 = fully curled, 180 = fully extended
    """
    mcp, pip, dip, tip = [landmarks[i] for i in finger_indices]
    
    # Calculate angles at PIP and DIP joints
    pip_angle = calculate_angle(mcp, pip, dip)
    dip_angle = calculate_angle(pip, dip, tip)
    
    # Average the angles (straight finger = ~180 degrees at each joint)
    avg_angle = (pip_angle + dip_angle) / 2
    
    # Map from joint angle (60-180) to servo angle (0-180)
    # 180 degrees (straight) -> 180 (open)
    # 60 degrees (curled) -> 0 (closed)
    servo_angle = int(np.interp(avg_angle, [60, 180], [0, 180]))
    return np.clip(servo_angle, 0, 180)

def calculate_thumb_curl(landmarks):
    """Calculate thumb curl - thumb has different geometry"""
    cmc = landmarks[1]
    mcp = landmarks[2]
    ip = landmarks[3]
    tip = landmarks[4]
    
    # Calculate angle at MCP and IP joints
    mcp_angle = calculate_angle(cmc, mcp, ip)
    ip_angle = calculate_angle(mcp, ip, tip)
    
    avg_angle = (mcp_angle + ip_angle) / 2
    servo_angle = int(np.interp(avg_angle, [60, 180], [0, 180]))
    return np.clip(servo_angle, 0, 180)

def get_all_finger_angles(landmarks):
    """Get curl angles for all fingers"""
    # Landmark indices for each finger: [MCP, PIP, DIP, TIP]
    finger_indices = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    
    angles = {}
    angles['thumb'] = calculate_thumb_curl(landmarks)
    angles['index'] = calculate_finger_curl(landmarks, finger_indices['index'])
    angles['middle'] = calculate_finger_curl(landmarks, finger_indices['middle'])
    angles['ring'] = calculate_finger_curl(landmarks, finger_indices['ring'])
    angles['pinky'] = calculate_finger_curl(landmarks, finger_indices['pinky'])
    
    return angles

def draw_landmarks_on_image(rgb_image, detection_result, angles):
    """Draw hand landmarks and angle info on the image"""
    hand_landmarks_list = detection_result.hand_landmarks
    
    if not hand_landmarks_list:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    h, w, _ = annotated_image.shape
    
    for hand_landmarks in hand_landmarks_list:
        landmarks_px = []
        for landmark in hand_landmarks:
            x_px = int(landmark.x * w)
            y_px = int(landmark.y * h)
            landmarks_px.append((x_px, y_px))
        
        # Draw connections with colors per finger
        finger_colors = [
            (255, 0, 255),   # Thumb - magenta
            (0, 255, 255),   # Index - cyan
            (0, 255, 0),     # Middle - green
            (255, 255, 0),   # Ring - yellow
            (255, 0, 0),     # Pinky - blue
        ]
        
        finger_connections = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],           # Thumb
            [(0, 5), (5, 6), (6, 7), (7, 8)],           # Index
            [(0, 9), (9, 10), (10, 11), (11, 12)],      # Middle
            [(0, 13), (13, 14), (14, 15), (15, 16)],    # Ring
            [(0, 17), (17, 18), (18, 19), (19, 20)],    # Pinky
        ]
        
        for finger_idx, connections in enumerate(finger_connections):
            color = finger_colors[finger_idx]
            for connection in connections:
                start = landmarks_px[connection[0]]
                end = landmarks_px[connection[1]]
                cv2.line(annotated_image, start, end, color, 3)
        
        # Draw palm connections
        palm_connections = [(5, 9), (9, 13), (13, 17)]
        for connection in palm_connections:
            start = landmarks_px[connection[0]]
            end = landmarks_px[connection[1]]
            cv2.line(annotated_image, start, end, (200, 200, 200), 2)
        
        # Draw landmarks
        for px in landmarks_px:
            cv2.circle(annotated_image, px, 5, (255, 255, 255), -1)
    
    # Draw angle info
    y_offset = 30
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    finger_keys = ['thumb', 'index', 'middle', 'ring', 'pinky']
    colors = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    
    for i, (name, key) in enumerate(zip(finger_names, finger_keys)):
        angle = angles.get(key, 0)
        bar_width = int(angle * 100 / 180)
        cv2.rectangle(annotated_image, (10, y_offset - 15), (10 + bar_width, y_offset + 5), colors[i], -1)
        cv2.rectangle(annotated_image, (10, y_offset - 15), (110, y_offset + 5), colors[i], 2)
        cv2.putText(annotated_image, f"{name}: {angle}", (120, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        y_offset += 35
    
    return annotated_image

def generate_frames():
    global finger_angles
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    last_angles = [180, 180, 180, 180, 180]
    smoothing = 0.3  # Lower = smoother but slower response
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        current_angles = {'thumb': 180, 'index': 180, 'middle': 180, 'ring': 180, 'pinky': 180}
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            current_angles = get_all_finger_angles(hand_landmarks)
            
            # Smooth the angles
            angle_list = [
                current_angles['thumb'],
                current_angles['index'],
                current_angles['middle'],
                current_angles['ring'],
                current_angles['pinky']
            ]
            
            smoothed = []
            for i, (new, old) in enumerate(zip(angle_list, last_angles)):
                smoothed_val = int(old + smoothing * (new - old))
                smoothed.append(smoothed_val)
            
            last_angles = smoothed
            
            # Update global state
            with lock:
                finger_angles['thumb'] = smoothed[0]
                finger_angles['index'] = smoothed[1]
                finger_angles['middle'] = smoothed[2]
                finger_angles['ring'] = smoothed[3]
                finger_angles['pinky'] = smoothed[4]
            
            # Send to Arduino
            send_all_angles(smoothed)
            
            # Update current_angles with smoothed values for display
            current_angles = {
                'thumb': smoothed[0],
                'index': smoothed[1],
                'middle': smoothed[2],
                'ring': smoothed[3],
                'pinky': smoothed[4]
            }
        
        # Draw landmarks
        rgb_frame = draw_landmarks_on_image(rgb_frame, detection_result, current_angles)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    detector.close()

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Finger Control</title>
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #1a1a2e;
                color: white;
            }
            h1 { color: #4CAF50; text-align: center; }
            .container {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                justify-content: center;
            }
            .video-container {
                border: 4px solid #4CAF50;
                border-radius: 10px;
                overflow: hidden;
            }
            .controls {
                background: #16213e;
                padding: 20px;
                border-radius: 10px;
                min-width: 300px;
            }
            .finger-control {
                margin: 15px 0;
            }
            .finger-control label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            .finger-control input[type="range"] {
                width: 100%;
                height: 20px;
            }
            .finger-control .value {
                text-align: right;
                font-size: 14px;
            }
            .thumb { color: #FF00FF; }
            .index { color: #00FFFF; }
            .middle { color: #00FF00; }
            .ring { color: #FFFF00; }
            .pinky { color: #FF0000; }
            
            .presets {
                margin-top: 20px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            button {
                padding: 12px;
                font-size: 16px;
                cursor: pointer;
                border: none;
                border-radius: 8px;
                color: white;
                background: #2196F3;
            }
            button:hover { opacity: 0.8; }
            .btn-open { background: #4CAF50; }
            .btn-close { background: #f44336; }
            
            .status-bar {
                margin-top: 15px;
                padding: 10px;
                background: #0f0f23;
                border-radius: 5px;
                font-family: monospace;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <h1>🤖 Finger-by-Finger Robot Control</h1>
        
        <div class="container">
            <div class="video-container">
                <img src="/video_feed" width="640" height="480">
            </div>
            
            <div class="controls">
                <h3>Manual Finger Control</h3>
                
                <div class="finger-control">
                    <label class="thumb">👍 Thumb: <span id="thumb-val">180</span>°</label>
                    <input type="range" id="thumb" min="0" max="180" value="180" 
                           oninput="sendFinger(0, this.value); document.getElementById('thumb-val').innerText=this.value">
                </div>
                
                <div class="finger-control">
                    <label class="index">☝️ Index: <span id="index-val">180</span>°</label>
                    <input type="range" id="index" min="0" max="180" value="180"
                           oninput="sendFinger(1, this.value); document.getElementById('index-val').innerText=this.value">
                </div>
                
                <div class="finger-control">
                    <label class="middle">🖕 Middle: <span id="middle-val">180</span>°</label>
                    <input type="range" id="middle" min="0" max="180" value="180"
                           oninput="sendFinger(2, this.value); document.getElementById('middle-val').innerText=this.value">
                </div>
                
                <div class="finger-control">
                    <label class="ring">💍 Ring: <span id="ring-val">180</span>°</label>
                    <input type="range" id="ring" min="0" max="180" value="180"
                           oninput="sendFinger(3, this.value); document.getElementById('ring-val').innerText=this.value">
                </div>
                
                <div class="finger-control">
                    <label class="pinky">🤙 Pinky: <span id="pinky-val">180</span>°</label>
                    <input type="range" id="pinky" min="0" max="180" value="180"
                           oninput="sendFinger(4, this.value); document.getElementById('pinky-val').innerText=this.value">
                </div>
                
                <div class="presets">
                    <button class="btn-open" onclick="preset('open')">✋ Open All</button>
                    <button class="btn-close" onclick="preset('close')">✊ Close All</button>
                    <button onclick="preset('point')">👆 Point</button>
                    <button onclick="preset('peace')">✌️ Peace</button>
                    <button onclick="preset('rock')">🤘 Rock</button>
                    <button onclick="preset('ok')">👌 OK</button>
                </div>
                
                <div class="status-bar">
                    <div>Arduino: <span id="arduino-status">Checking...</span></div>
                    <div>Detected: T:<span id="d-thumb">-</span> I:<span id="d-index">-</span> M:<span id="d-middle">-</span> R:<span id="d-ring">-</span> P:<span id="d-pinky">-</span></div>
                </div>
            </div>
        </div>
        
        <script>
            async function sendFinger(finger, angle) {
                await fetch(`/finger/${finger}/${angle}`);
            }
            
            async function preset(name) {
                const presets = {
                    'open': [180, 180, 180, 180, 180],
                    'close': [0, 0, 0, 0, 0],
                    'point': [0, 180, 0, 0, 0],
                    'peace': [0, 180, 180, 0, 0],
                    'rock': [0, 180, 0, 0, 180],
                    'ok': [180, 0, 180, 180, 180]
                };
                
                const angles = presets[name];
                if (angles) {
                    await fetch(`/all/${angles.join(',')}`);
                    updateSliders(angles);
                }
            }
            
            function updateSliders(angles) {
                const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
                fingers.forEach((f, i) => {
                    document.getElementById(f).value = angles[i];
                    document.getElementById(f + '-val').innerText = angles[i];
                });
            }
            
            async function updateStatus() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    
                    document.getElementById('arduino-status').innerText = 
                        data.arduino_connected ? '✅ Connected' : '❌ Disconnected';
                    
                    document.getElementById('d-thumb').innerText = data.angles.thumb;
                    document.getElementById('d-index').innerText = data.angles.index;
                    document.getElementById('d-middle').innerText = data.angles.middle;
                    document.getElementById('d-ring').innerText = data.angles.ring;
                    document.getElementById('d-pinky').innerText = data.angles.pinky;
                } catch (e) {
                    console.error(e);
                }
            }
            
            setInterval(updateStatus, 200);
        </script>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finger/<int:finger>/<int:angle>')
def set_finger(finger, angle):
    angle = max(0, min(180, angle))
    send_finger_angle(finger, angle)
    return jsonify({'success': True, 'finger': finger, 'angle': angle})

@app.route('/all/<angles>')
def set_all(angles):
    try:
        angle_list = [int(a) for a in angles.split(',')]
        if len(angle_list) == 5:
            send_all_angles(angle_list)
            return jsonify({'success': True, 'angles': angle_list})
    except:
        pass
    return jsonify({'success': False})

@app.route('/status')
def status():
    with lock:
        angles = finger_angles.copy()
    return jsonify({
        'angles': angles,
        'arduino_connected': arduino is not None and arduino.is_open
    })

if __name__ == '__main__':
    download_model()
    connect_arduino()
    print("Starting server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
