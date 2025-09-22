


from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
from flask_mail import Mail, Message
import os
import cv2
import json
import base64
import serial
import serial.tools.list_ports
from io import BytesIO
from ultralytics import YOLO
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

# Configure Flask-Mail with .env settings
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 465))
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'False') == 'True'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'True') == 'True'
mail = Mail(app)

# Connect to MongoDB
client = MongoClient(os.getenv('MONGO_URI'))
db = client['plants']
plants_collection = db['plantlocations']

# Load YOLO model
MODEL_PATH = "my_model.pt"
model = YOLO(MODEL_PATH)

# Find Arduino port
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "Arduino" in port.description or "CH340" in port.description or "USB Serial" in port.description:
            return port.device
    return None

# Automatically detect Arduino port
arduino_port = find_arduino_port()

if arduino_port:
    try:
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        print(f"Connected to Arduino on {arduino_port}")
    except Exception as e:
        arduino = None
        print(f"Could not connect to Arduino: {e}")
else:
    arduino = None
    print("No Arduino found")

# Video capture
camera = cv2.VideoCapture(0)
CAPTURES_DIR = "static/captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def send_plant_detection_email(doc):
    """Send email notification with plant detection data from MongoDB"""
    image_id = str(doc["_id"])
    location = doc["location"]
    detected_plants = doc["detected_plants"]
    timestamp = doc["timestamp"]
    
    subject = f"Plant Detection Report: {len(detected_plants)} plants found"
    
    # Create message body with detected plants and location
    plant_list = "\n".join([f"- {plant['name']} (Confidence: {plant['confidence']:.2%})" 
                           for plant in detected_plants])
    
    body = f"""Plants detected at location:
Latitude: {location['latitude']}
Longitude: {location['longitude']}
Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Detected Plants:
{plant_list}

Image ID: {image_id}
"""
    
    msg = Message(
        subject=subject,
        sender=app.config['MAIL_USERNAME'],
        recipients=['rajesh.duvvuru@vitap.ac.in'],  # You can make this configurable
        body=body
    )
    
    # Create and attach the image if it's not already saved
    temp_image_path = os.path.join(CAPTURES_DIR, f"{image_id}.jpg")
    if not os.path.exists(temp_image_path):
        # Decode base64 image and save to temporary file
        img_data = base64.b64decode(doc["image"])
        with open(temp_image_path, "wb") as f:
            f.write(img_data)
    
    # Attach the image to the email
    with app.open_resource(temp_image_path) as fp:
        msg.attach(f"plant_detection_{image_id}.jpg", "image/jpeg", fp.read())
    
    mail.send(msg)
    return True

@app.route('/capture_analyze', methods=['POST'])
def capture_analyze():
    data = request.json
    latitude = str(data['latitude'])
    longitude = str(data['longitude'])
    email_notification = data.get('email_notification', False)  # Optional parameter
    
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    detected_plants = []
    
    for detection in results[0].boxes:
        confidence = detection.conf.item()
        if confidence < 0.5:
            continue  # Ignore low-confidence detections
            
        xyxy = detection.xyxy.cpu().numpy().astype(int)
        class_idx = int(detection.cls.item())
        
        # Fetch class name from model
        plant_name = model.names[class_idx]
        detected_plants.append({
            "name": plant_name,
            "confidence": float(confidence)
        })
        
        label = f"{plant_name}: {int(confidence * 100)}%"
        
        # Draw bounding box
        cv2.rectangle(frame, (xyxy[0][0], xyxy[0][1]), (xyxy[0][2], xyxy[0][3]), (0, 255, 0), 2)
        
        # Put label text
        cv2.putText(frame, label, (xyxy[0][0], xyxy[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Only proceed with MongoDB storage if plants were detected
    if not detected_plants:
        # Create a temporary image to show "No plants detected"
        cv2.putText(frame, "No plants detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        temp_no_plant_path = os.path.join(CAPTURES_DIR, f"no_plant_{datetime.now().timestamp()}.jpg")
        cv2.imwrite(temp_no_plant_path, frame)
        return jsonify({"image_path": temp_no_plant_path, "no_plants_detected": True})
    
    # Convert image to base64 for MongoDB storage
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create document for MongoDB
    document = {
        "location": {
            "latitude": float(latitude),
            "longitude": float(longitude)
        },
        "image": img_base64,
        "detected_plants": detected_plants,
        "timestamp": datetime.now()
    }
    
    # Insert into MongoDB
    result = plants_collection.insert_one(document)
    document["_id"] = result.inserted_id  # Add the new ID to the document
    
    # Save a temporary copy for display
    temp_image_path = os.path.join(CAPTURES_DIR, f"{result.inserted_id}.jpg")
    cv2.imwrite(temp_image_path, frame)
    
    # Send email notification if requested
    email_sent = False
    if email_notification:
        try:
            email_sent = send_plant_detection_email(document)
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    return jsonify({
        "image_path": temp_image_path, 
        "image_id": str(result.inserted_id),
        "plants_detected": True,
        "plant_count": len(detected_plants),
        "email_sent": email_sent
    })

@app.route('/send_db_email', methods=['POST'])
def send_db_email():
    """Send email with data from a specific MongoDB document"""
    data = request.json
    image_id = data.get('image_id')
    recipient = data.get('recipient')
    
    # Validate email is provided
    if not recipient or '@' not in recipient:
        return jsonify({"error": "Valid recipient email is required"}), 400
    
    if image_id:
        # Send email for a specific image
        try:
            doc = plants_collection.find_one({"_id": ObjectId(image_id)})
            if not doc:
                return jsonify({"error": "Image not found"}), 404
            
            msg = create_email_from_doc(doc, recipient)
            mail.send(msg)
            
            # Log email sent
            app.logger.info(f"Email sent to {recipient} for image {image_id}")
            
            return jsonify({"status": "success", "message": f"Email sent for image {image_id}"})
        except Exception as e:
            app.logger.error(f"Error sending email: {str(e)}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Image ID is required"}), 400

def create_email_from_doc(doc, recipient):
    """Create an email message from a MongoDB document"""
    image_id = str(doc["_id"])
    location = doc["location"]
    detected_plants = doc["detected_plants"]
    timestamp = doc["timestamp"]
    
    subject = f"Plant Detection Report: {len(detected_plants)} plants found"
    
    # Create message body with detected plants and location
    plant_list = "\n".join([f"- {plant['name']} (Confidence: {plant['confidence']:.2%})"
                           for plant in detected_plants])
    
    body = f"""Plants detected at location: 
Latitude: {location['latitude']} 
Longitude: {location['longitude']}
Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Detected Plants:
{plant_list}

Image ID: {image_id}
"""
    
    msg = Message(
        subject=subject,
        sender=app.config['MAIL_USERNAME'],
        recipients=[recipient],
        body=body
    )
    
    # Create and attach the image if it's not already saved
    temp_image_path = os.path.join(CAPTURES_DIR, f"{image_id}.jpg")
    if not os.path.exists(temp_image_path):
        # Decode base64 image and save to temporary file
        img_data = base64.b64decode(doc["image"])
        with open(temp_image_path, "wb") as f:
            f.write(img_data)
    
    # Attach the image to the email
    with app.open_resource(temp_image_path) as fp:
        msg.attach(f"plant_detection_{image_id}.jpg", "image/jpeg", fp.read())
    return msg

@app.route('/send_location_report', methods=['POST'])
def send_location_report():
    """Send email report for all plants detected at a specific location"""
    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    proximity_range = data.get('range', 0.0005)  # Default ~50 meters
    recipient = data.get('recipient', 'rajesh.duvvuru@vitap.ac.in')
    
    if not latitude or not longitude:
        return jsonify({"error": "Latitude and longitude are required"}), 400
    
    try:
        # Query MongoDB for images at this location
        query = {
            "location.latitude": {"$gte": float(latitude) - proximity_range, 
                                 "$lte": float(latitude) + proximity_range},
            "location.longitude": {"$gte": float(longitude) - proximity_range, 
                                  "$lte": float(longitude) + proximity_range}
        }
        
        cursor = plants_collection.find(query).sort("timestamp", -1)  # Sort by most recent first
        docs = list(cursor)
        
        if not docs:
            return jsonify({"error": "No plant detections found at this location"}), 404
        
        # Create email with summary of all detections
        subject = f"Plant Detection Summary Report: {len(docs)} detections"
        
        body = f"""Plant Detection Summary for Location:
Latitude: {latitude}
Longitude: {longitude}
Search Range: {proximity_range} degrees

Total Detections: {len(docs)}
Time Range: {docs[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} to {docs[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Detailed Reports:
"""
        
        # Create a summary of all detections
        unique_plants = {}
        for doc in docs:
            for plant in doc['detected_plants']:
                plant_name = plant['name']
                if plant_name in unique_plants:
                    unique_plants[plant_name] += 1
                else:
                    unique_plants[plant_name] = 1
                    
        body += "\nPlant Species Summary:\n"
        for plant_name, count in unique_plants.items():
            body += f"- {plant_name}: {count} detections\n"
            
        body += "\n\nDetailed Detection Reports:\n"
        
        # Add individual detection reports (limit to first 10 to avoid huge emails)
        max_reports = min(10, len(docs))
        for i in range(max_reports):
            doc = docs[i]
            plant_list = ", ".join([plant['name'] for plant in doc['detected_plants']])
            body += f"\n{i+1}. Detection on {doc['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            body += f"   Plants: {plant_list}\n"
            body += f"   Image ID: {doc['_id']}\n"
            
        if len(docs) > max_reports:
            body += f"\n(Showing {max_reports} of {len(docs)} total detections)\n"
            
        msg = Message(
            subject=subject,
            sender=app.config['MAIL_USERNAME'],
            recipients=[recipient],
            body=body
        )
        
        # Attach the most recent image
        most_recent_doc = docs[0]
        image_id = str(most_recent_doc["_id"])
        temp_image_path = os.path.join(CAPTURES_DIR, f"{image_id}.jpg")
        
        if not os.path.exists(temp_image_path):
            # Decode base64 image and save to temporary file
            img_data = base64.b64decode(most_recent_doc["image"])
            with open(temp_image_path, "wb") as f:
                f.write(img_data)
        
        with app.open_resource(temp_image_path) as fp:
            msg.attach(f"latest_detection.jpg", "image/jpeg", fp.read())
            
        mail.send(msg)
        return jsonify({"status": "success", "message": f"Location report sent with {len(docs)} detections"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send_all_reports', methods=['POST'])
def send_all_reports():
    """Send a complete report of all plant detections in the database"""
    data = request.json
    recipient = data.get('recipient', 'rajesh.duvvuru@vitap.ac.in')
    max_days = data.get('max_days')  # Optional: limit to recent days
    
    try:
        # Build query
        query = {}
        if max_days:
            # Limit to records from the last X days
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=int(max_days))
            query["timestamp"] = {"$gte": cutoff_date}
        
        # Get all documents
        cursor = plants_collection.find(query).sort("timestamp", -1)
        docs = list(cursor)
        
        if not docs:
            return jsonify({"error": "No plant detections found in database"}), 404
        
        # Create email with summary of all detections
        time_range = ""
        if max_days:
            time_range = f" (Last {max_days} days)"
            
        subject = f"Complete Plant Detection Database Report{time_range}: {len(docs)} records"
        
        body = f"""Plant Detection Database Report{time_range}

Total Records: {len(docs)}
Time Range: {docs[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} to {docs[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Summary:
"""
        
        # Count detections by location
        locations = {}
        unique_plants = {}
        
        for doc in docs:
            # Group by rounded coordinates for location clustering
            lat_rounded = round(doc['location']['latitude'], 3)
            lng_rounded = round(doc['location']['longitude'], 3)
            location_key = f"{lat_rounded},{lng_rounded}"
            
            if location_key in locations:
                locations[location_key]['count'] += 1
            else:
                locations[location_key] = {
                    'lat': doc['location']['latitude'],
                    'lng': doc['location']['longitude'],
                    'count': 1
                }
                
            # Count plant species
            for plant in doc['detected_plants']:
                plant_name = plant['name']
                if plant_name in unique_plants:
                    unique_plants[plant_name] += 1
                else:
                    unique_plants[plant_name] = 1
                    
        # Add location summary
        body += f"\nDetections by Location ({len(locations)} unique locations):\n"
        for location_key, data in sorted(locations.items(), key=lambda x: x[1]['count'], reverse=True):
            body += f"- Lat: {data['lat']:.5f}, Lng: {data['lng']:.5f}: {data['count']} detections\n"
            
        # Add plant species summary
        body += f"\nPlant Species Summary ({len(unique_plants)} unique species):\n"
        for plant_name, count in sorted(unique_plants.items(), key=lambda x: x[1], reverse=True):
            body += f"- {plant_name}: {count} detections\n"
            
        msg = Message(
            subject=subject,
            sender=app.config['MAIL_USERNAME'],
            recipients=[recipient],
            body=body
        )
        
        mail.send(msg)
        return jsonify({
            "status": "success", 
            "message": f"Complete database report sent with {len(docs)} records"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_images')
def get_images():
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))
    
    # Set a proximity range for finding images (approximately within 50 meters)
    proximity_range = 0.0005  # Roughly 50 meters in decimal degrees
    
    # Query MongoDB for images at this location
    query = {
        "location.latitude": {"$gte": latitude - proximity_range, "$lte": latitude + proximity_range},
        "location.longitude": {"$gte": longitude - proximity_range, "$lte": longitude + proximity_range}
    }
    
    cursor = plants_collection.find(query).sort("timestamp", -1)  # Sort by most recent first
    
    images = []
    for doc in cursor:
        # Create temporary file for each image for display
        img_id = str(doc["_id"])
        temp_path = os.path.join(CAPTURES_DIR, f"{img_id}.jpg")
        
        # Check if temporary file already exists
        if not os.path.exists(temp_path):
            # Decode base64 image and save to temporary file
            img_data = base64.b64decode(doc["image"])
            with open(temp_path, "wb") as f:
                f.write(img_data)
        
        images.append(f"/{CAPTURES_DIR}/{img_id}.jpg")
    
    return jsonify(images)

@app.route('/get_image_details/<image_id>')
def get_image_details(image_id):
    try:
        doc = plants_collection.find_one({"_id": ObjectId(image_id)})
        if doc:
            return jsonify({
                "latitude": doc["location"]["latitude"],
                "longitude": doc["location"]["longitude"],
                "detected_plants": doc["detected_plants"],
                "timestamp": doc["timestamp"].isoformat()
            })
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup_temp_files', methods=['POST'])
def cleanup_temp_files():
    """Clean up temporary image files to save disk space"""
    try:
        for file in os.listdir(CAPTURES_DIR):
            os.remove(os.path.join(CAPTURES_DIR, file))
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Cleanup temporary files on startup
for file in os.listdir(CAPTURES_DIR):
    try:
        os.remove(os.path.join(CAPTURES_DIR, file))
    except:
        pass
    
@socketio.on('move')
def move(direction):
    if arduino:
        arduino.write(direction.encode())  # Send command to Arduino
        print(f"Sent to Arduino: {direction}")

@app.route('/delete_image/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    try:
        # Find the image in MongoDB
        result = plants_collection.delete_one({"_id": ObjectId(image_id)})
        
        if result.deleted_count == 0:
            return jsonify({"error": "Image not found"}), 404
            
        # Delete the temporary file if it exists
        temp_path = os.path.join(CAPTURES_DIR, f"{image_id}.jpg")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_all_images', methods=['DELETE'])
def delete_all_images():
    try:
        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        
        # Set a proximity range for finding images (approximately within 50 meters)
        proximity_range = 0.0005  # Roughly 50 meters in decimal degrees
        
        # Query MongoDB for images at this location
        query = {
            "location.latitude": {"$gte": latitude - proximity_range, "$lte": latitude + proximity_range},
            "location.longitude": {"$gte": longitude - proximity_range, "$lte": longitude + proximity_range}
        }
        
        # Find all matching documents and collect their IDs
        image_ids = []
        cursor = plants_collection.find(query, {"_id": 1})
        for doc in cursor:
            image_ids.append(str(doc["_id"]))
        
        # Delete all matching documents
        result = plants_collection.delete_many(query)
        
        # Delete all temporary files
        for image_id in image_ids:
            temp_path = os.path.join(CAPTURES_DIR, f"{image_id}.jpg")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return jsonify({"status": "success", "deleted_count": result.deleted_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send_test_email')
def send_test_email():
    try:
        msg = Message(
            subject='Plant Detection System Test',
            sender=app.config['MAIL_USERNAME'],
            recipients=['rajesh.duvvuru@vitap.ac.in'],
            body='This is a test email from your Plant Detection System. If you received this, the email functionality is working correctly.'
        )
        mail.send(msg)
        return jsonify({"status": "success", "message": "Test email sent successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)