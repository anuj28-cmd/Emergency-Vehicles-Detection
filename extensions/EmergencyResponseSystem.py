import cv2
import numpy as np
import time
import threading
import queue
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class CameraNode:
    """Represents a single camera in the emergency response network"""
    
    def __init__(self, camera_id, location_name, model_path='emergency_vehicle_model.h5'):
        self.camera_id = camera_id
        self.location_name = location_name
        self.model = load_model(model_path)
        self.class_names = {0: 'Emergency Vehicle', 1: 'Normal Vehicle'}
        self.last_detection_time = 0
        self.detection_cooldown = 5  # seconds
        self.emergency_detected = False
        self.emergency_confidence = 0
        self.confidence_threshold = 0.70
    
    def process_frame(self, frame):
        """Process a frame and detect emergency vehicles"""
        # Resize frame for model input
        input_img = cv2.resize(frame, (224, 224))
        x = image.img_to_array(input_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        preds = self.model.predict(x, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx] * 100
        
        # Check for emergency vehicle with high confidence
        current_time = time.time()
        if class_idx == 0 and confidence > self.confidence_threshold * 100:
            self.emergency_detected = True
            self.emergency_confidence = confidence
            self.last_detection_time = current_time
        elif current_time - self.last_detection_time > self.detection_cooldown:
            self.emergency_detected = False
            self.emergency_confidence = 0
            
        # Augment video frame with detection info
        self._annotate_frame(frame, class_idx, confidence)
        
        return frame, self.emergency_detected, self.emergency_confidence
    
    def _annotate_frame(self, frame, class_idx, confidence):
        """Add visual information to the frame"""
        # Add camera ID and location
        cv2.putText(frame, f"Camera {self.camera_id}: {self.location_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection result
        class_name = self.class_names[class_idx]
        result_text = f"{class_name}: {confidence:.2f}%"
        color = (0, 0, 255) if class_idx == 0 else (0, 255, 0)
        cv2.putText(frame, result_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add alert if emergency vehicle detected
        if self.emergency_detected:
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED!", 
                      (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.9, (0, 0, 255), 2)
            # Add red border around frame
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                         (0, 0, 255), 5)

class EmergencyResponseSystem:
    """Multi-camera system to track emergency vehicles across locations"""
    
    def __init__(self, model_path='emergency_vehicle_model.h5'):
        self.camera_nodes = {}
        self.alerts_queue = queue.Queue()
        self.model_path = model_path
        self.running = False
        self.emergency_tracker = {}  # Track emergency vehicles across cameras
    
    def add_camera(self, camera_id, location_name, video_source):
        """Add a camera to the monitoring system"""
        self.camera_nodes[camera_id] = {
            'node': CameraNode(camera_id, location_name, self.model_path),
            'source': video_source,
            'last_alert': 0
        }
    
    def start_monitoring(self):
        """Start monitoring all cameras"""
        self.running = True
        
        # Start a processing thread for each camera
        for camera_id, camera_info in self.camera_nodes.items():
            thread = threading.Thread(
                target=self._process_camera_feed,
                args=(camera_id, camera_info),
                daemon=True
            )
            thread.start()
        
        # Start alert processing thread
        alert_thread = threading.Thread(
            target=self._process_alerts,
            daemon=True
        )
        alert_thread.start()
        
        # Display all camera feeds
        self._display_feeds()
    
    def _process_camera_feed(self, camera_id, camera_info):
        """Process video from a specific camera"""
        node = camera_info['node']
        cap = cv2.VideoCapture(camera_info['source'])
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Try to reconnect if feed is lost
                time.sleep(1)
                cap = cv2.VideoCapture(camera_info['source'])
                continue
            
            # Process the frame
            processed_frame, emergency_detected, confidence = node.process_frame(frame)
            
            # Store processed frame for display
            camera_info['frame'] = processed_frame
            
            # Generate alert if emergency vehicle detected
            if emergency_detected:
                current_time = time.time()
                if current_time - camera_info['last_alert'] > 3:  # Alert cooldown
                    self.alerts_queue.put({
                        'camera_id': camera_id,
                        'location': node.location_name,
                        'confidence': confidence,
                        'timestamp': current_time,
                        'frame': processed_frame.copy()
                    })
                    camera_info['last_alert'] = current_time
                    
                    # Update emergency tracker
                    self._update_emergency_tracker(camera_id, node.location_name, confidence)
            
            time.sleep(0.01)  # Small delay to reduce CPU usage
        
        cap.release()
    
    def _update_emergency_tracker(self, camera_id, location, confidence):
        """Track emergency vehicles moving between cameras"""
        current_time = time.time()
        
        # Add new detection to tracker
        self.emergency_tracker[current_time] = {
            'camera_id': camera_id,
            'location': location,
            'confidence': confidence
        }
        
        # Clean up old detections (older than 2 minutes)
        self.emergency_tracker = {
            k: v for k, v in self.emergency_tracker.items()
            if current_time - k < 120
        }
        
        # Analyze movement pattern
        self._analyze_emergency_movement()
    
    def _analyze_emergency_movement(self):
        """Analyze emergency vehicle movement patterns"""
        # Sort detections by time
        detections = sorted(self.emergency_tracker.items())
        
        # Need at least 2 detections to analyze movement
        if len(detections) < 2:
            return
        
        # Get the two most recent detections
        latest = detections[-1][1]
        previous = detections[-2][1]
        
        # If detected at different cameras, estimate movement direction
        if latest['camera_id'] != previous['camera_id']:
            print(f"Emergency vehicle moving from {previous['location']} to {latest['location']}")
    
    def _process_alerts(self):
        """Process emergency alerts"""
        while self.running:
            try:
                alert = self.alerts_queue.get(timeout=0.5)
                print(f"ALERT! Emergency vehicle detected at {alert['location']} "
                     f"(Camera {alert['camera_id']}) with {alert['confidence']:.2f}% confidence")
                
                # Save the alert frame
                alert_filename = f"alert_{alert['camera_id']}_{int(alert['timestamp'])}.jpg"
                cv2.imwrite(alert_filename, alert['frame'])
                print(f"Alert image saved as {alert_filename}")
                
                self.alerts_queue.task_done()
            except queue.Empty:
                pass
    
    def _display_feeds(self):
        """Display all camera feeds in a grid"""
        while self.running:
            if not self.camera_nodes:
                time.sleep(0.1)
                continue
                
            # Count number of cameras to determine grid size
            n_cameras = len(self.camera_nodes)
            grid_size = int(np.ceil(np.sqrt(n_cameras)))
            
            # Create grid to display all feeds
            grid_height = 480
            grid_width = 640
            grid = np.zeros((grid_height * grid_size, grid_width * grid_size, 3), dtype=np.uint8)
            
            # Populate grid with camera feeds
            i = 0
            for camera_id, camera_info in self.camera_nodes.items():
                if 'frame' not in camera_info:
                    continue
                    
                frame = camera_info['frame']
                if frame is None:
                    continue
                    
                # Resize frame to fit in grid
                resized = cv2.resize(frame, (grid_width, grid_height))
                
                # Calculate position in grid
                row = i // grid_size
                col = i % grid_size
                
                # Place in grid
                y_start = row * grid_height
                y_end = y_start + grid_height
                x_start = col * grid_width
                x_end = x_start + grid_width
                
                grid[y_start:y_end, x_start:x_end] = resized
                i += 1
            
            # Display grid
            cv2.imshow('Emergency Response System - All Cameras', grid)
            
            # Check for exit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()

# Example usage
def run_emergency_response_system():
    system = EmergencyResponseSystem()
    
    # Add cameras (in real system these would be network camera streams or video files)
    # For demo purposes, we're using the webcam multiple times
    system.add_camera(1, "Main Street", 0)  # Using webcam for demo
    
    # If you have test videos, you could use them like this:
    # system.add_camera(2, "Highway Junction", "test_video1.mp4")
    # system.add_camera(3, "Hospital Entrance", "test_video2.mp4")
    
    print("Starting Emergency Response System...")
    print("Press 'q' to quit")
    system.start_monitoring()

if __name__ == "__main__":
    run_emergency_response_system()