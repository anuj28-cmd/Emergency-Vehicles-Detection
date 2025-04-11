import cv2
import numpy as np
import time
import heapq
import folium
import webbrowser
import os
from threading import Thread
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class TrafficNode:
    """Represents a traffic junction with congestion detection"""
    def __init__(self, node_id, name, latitude, longitude):
        self.node_id = node_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.traffic_level = 0.0  # 0.0 = free flowing, 1.0 = complete gridlock
        self.has_emergency_vehicle = False
        self.last_update = time.time()
    
    def update_traffic(self, traffic_level):
        """Update traffic congestion level (0.0 to 1.0)"""
        self.traffic_level = max(0.0, min(1.0, traffic_level))
        self.last_update = time.time()
    
    def set_emergency_vehicle(self, detected):
        """Mark if this node has an emergency vehicle present"""
        self.has_emergency_vehicle = detected
        if detected:
            # Reduce traffic level if emergency vehicle is detected
            # (assuming other vehicles are making way)
            self.traffic_level = max(0.0, self.traffic_level - 0.3)
    
    def get_travel_time(self, base_time):
        """Calculate travel time through this node based on congestion"""
        # More congestion = longer travel time
        congestion_factor = 1.0 + (self.traffic_level * 5.0)  # Up to 6x slower in gridlock
        return base_time * congestion_factor

class TrafficEdge:
    """Represents a road between two traffic nodes"""
    def __init__(self, edge_id, start_node, end_node, distance, speed_limit):
        self.edge_id = edge_id
        self.start_node = start_node  # Node ID
        self.end_node = end_node      # Node ID
        self.distance = distance      # in km
        self.speed_limit = speed_limit  # in km/h
        self.base_travel_time = (distance / speed_limit) * 60  # in minutes
        self.traffic_level = 0.0  # 0.0 = free flowing, 1.0 = complete gridlock
    
    def update_traffic(self, traffic_level):
        """Update traffic congestion level on this road segment"""
        self.traffic_level = max(0.0, min(1.0, traffic_level))
    
    def get_travel_time(self):
        """Calculate current travel time based on traffic conditions"""
        # More congestion = longer travel time
        congestion_factor = 1.0 + (self.traffic_level * 4.0)  # Up to 5x slower in gridlock
        return self.base_travel_time * congestion_factor

class RouteOptimizer:
    """Optimizes routes for emergency vehicles based on traffic conditions"""
    def __init__(self, model_path='emergency_vehicle_model.h5'):
        # Load the emergency vehicle detection model
        self.model = load_model(model_path)
        self.class_names = {0: 'Emergency Vehicle', 1: 'Normal Vehicle'}
        
        # Traffic network
        self.nodes = {}  # Dictionary of TrafficNode objects by ID
        self.edges = {}  # Dictionary of TrafficEdge objects by ID
        self.adjacency = {}  # Adjacency list for graph algorithms
        
        # Camera input sources
        self.cameras = {}  # Dictionary mapping camera IDs to node IDs
        
        # City coordinates (for map display)
        self.center_lat = 12.9716  # Default to Bangalore, India
        self.center_lng = 77.5946
    
    def add_node(self, node_id, name, latitude, longitude):
        """Add a traffic junction node to the network"""
        self.nodes[node_id] = TrafficNode(node_id, name, latitude, longitude)
        self.adjacency[node_id] = []  # Initialize adjacency list
        
        # Update center coordinates
        if len(self.nodes) == 1:
            # First node sets the center
            self.center_lat = latitude
            self.center_lng = longitude
        else:
            # Update center as the average of all nodes
            all_lats = [node.latitude for node in self.nodes.values()]
            all_lngs = [node.longitude for node in self.nodes.values()]
            self.center_lat = sum(all_lats) / len(all_lats)
            self.center_lng = sum(all_lngs) / len(all_lngs)
    
    def add_edge(self, edge_id, start_node_id, end_node_id, distance, speed_limit):
        """Add a road segment between two junctions"""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            print(f"Error: Cannot add edge {edge_id}. Node(s) not found.")
            return False
        
        # Create edge
        edge = TrafficEdge(edge_id, start_node_id, end_node_id, distance, speed_limit)
        self.edges[edge_id] = edge
        
        # Update adjacency list (assuming bidirectional roads)
        self.adjacency[start_node_id].append((end_node_id, edge_id))
        self.adjacency[end_node_id].append((start_node_id, edge_id))
        
        return True
    
    def add_camera(self, camera_id, video_source, node_id):
        """Associate a camera with a specific traffic node"""
        if node_id not in self.nodes:
            print(f"Error: Cannot add camera {camera_id}. Node {node_id} not found.")
            return False
        
        self.cameras[camera_id] = {
            'node_id': node_id,
            'source': video_source
        }
        
        return True
    
    def start_monitoring(self):
        """Start monitoring all cameras to detect traffic conditions"""
        print("Starting traffic monitoring...")
        
        # Start a thread for each camera
        for camera_id, camera_info in self.cameras.items():
            thread = Thread(
                target=self._process_camera_feed,
                args=(camera_id, camera_info),
                daemon=True
            )
            thread.start()
        
        # Start a thread to regularly update the map
        map_thread = Thread(
            target=self._update_map_periodically,
            daemon=True
        )
        map_thread.start()
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping monitoring...")
    
    def _process_camera_feed(self, camera_id, camera_info):
        """Process video from a traffic camera to estimate congestion and detect emergency vehicles"""
        node_id = camera_info['node_id']
        cap = cv2.VideoCapture(camera_info['source'])
        
        # Set up display window
        window_name = f"Traffic Camera {camera_id}: {self.nodes[node_id].name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # For demo purposes, let's randomly vary traffic levels
        import random
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read from camera {camera_id}")
                time.sleep(2)
                cap = cv2.VideoCapture(camera_info['source'])
                continue
            
            # Process frame to detect vehicles
            try:
                emergency_detected = self._detect_emergency_vehicle(frame)
                
                # Update node with emergency vehicle status
                self.nodes[node_id].set_emergency_vehicle(emergency_detected)
                
                # In a real system, we would analyze the frame to estimate congestion
                # For this demo, we'll simulate varying traffic levels
                if random.random() < 0.05:  # Occasionally update traffic level
                    traffic_level = random.uniform(0.0, 0.9)
                    self.nodes[node_id].update_traffic(traffic_level)
                    
                    # Also update adjacent edges with similar traffic levels (slightly randomized)
                    for _, edge_id in self.adjacency[node_id]:
                        edge_traffic = max(0, min(1.0, traffic_level + random.uniform(-0.2, 0.2)))
                        self.edges[edge_id].update_traffic(edge_traffic)
                
                # Add traffic information to display
                self._annotate_traffic_frame(frame, node_id)
                
                # Show the frame
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error processing frame from camera {camera_id}: {e}")
            
            time.sleep(0.03)  # ~30 fps
        
        cap.release()
        cv2.destroyWindow(window_name)
    
    def _detect_emergency_vehicle(self, frame):
        """Detect if an emergency vehicle is present in the frame"""
        # Resize frame for model input
        input_img = cv2.resize(frame, (224, 224))
        x = image.img_to_array(input_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        preds = self.model.predict(x, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]
        
        # Return True if emergency vehicle detected with high confidence
        return class_idx == 0 and confidence > 0.7
    
    def _annotate_traffic_frame(self, frame, node_id):
        """Add traffic information overlay to the frame"""
        node = self.nodes[node_id]
        
        # Add node information
        cv2.putText(frame, f"Junction: {node.name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add traffic level
        traffic_text = f"Traffic Level: {node.traffic_level:.2f}"
        color = self._get_traffic_color(node.traffic_level)
        cv2.putText(frame, traffic_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add emergency vehicle status if detected
        if node.has_emergency_vehicle:
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED", 
                      (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.9, (0, 0, 255), 2)
            # Add red border
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                         (0, 0, 255), 5)
    
    def _get_traffic_color(self, traffic_level):
        """Get color based on traffic level (green=free flowing, red=gridlock)"""
        if traffic_level < 0.3:
            return (0, 255, 0)  # Green
        elif traffic_level < 0.6:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def find_optimal_route(self, start_node_id, end_node_id):
        """Find the fastest route given current traffic conditions using Dijkstra's algorithm"""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return None, None
        
        # Initialize data structures for Dijkstra's algorithm
        distances = {node_id: float('infinity') for node_id in self.nodes}
        distances[start_node_id] = 0
        previous = {node_id: None for node_id in self.nodes}
        visited = set()
        
        # Priority queue for Dijkstra's algorithm
        pq = [(0, start_node_id)]
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_node == end_node_id:
                break
                
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Check all neighbors
            for neighbor, edge_id in self.adjacency[current_node]:
                if neighbor in visited:
                    continue
                    
                edge = self.edges[edge_id]
                
                # Calculate travel time considering traffic
                if edge.start_node == current_node:
                    next_node = edge.end_node
                else:
                    next_node = edge.start_node
                    
                travel_time = edge.get_travel_time()
                
                # Add node traversal time
                travel_time += self.nodes[next_node].get_travel_time(0.5)  # 0.5 minute base time for junction
                
                # Update distance if we found a shorter path
                new_distance = distances[current_node] + travel_time
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = (current_node, edge_id)
                    heapq.heappush(pq, (new_distance, neighbor))
        
        # Reconstruct path
        path = []
        edges = []
        current = end_node_id
        
        while current != start_node_id:
            if previous[current] is None:
                # No path found
                return None, None
                
            prev_node, edge_id = previous[current]
            path.insert(0, current)
            edges.insert(0, edge_id)
            current = prev_node
            
        path.insert(0, start_node_id)
        
        return path, edges
    
    def _update_map_periodically(self, interval=10):
        """Update the traffic map periodically"""
        while True:
            self.generate_traffic_map()
            time.sleep(interval)
    
    def generate_traffic_map(self, route=None):
        """Generate an HTML map with traffic conditions and optional route"""
        # Create a map centered at the average location of all nodes
        m = folium.Map(location=[self.center_lat, self.center_lng], zoom_start=13)
        
        # Add all nodes to the map
        for node_id, node in self.nodes.items():
            # Set color based on traffic level
            if node.has_emergency_vehicle:
                color = 'red'
                icon = 'ambulance'
                prefix = 'fa'
            else:
                color = self._get_folium_color(node.traffic_level)
                icon = 'map-marker'
                prefix = 'fa'
            
            # Create popup with node information
            popup_html = f"""
            <b>{node.name}</b><br>
            Traffic Level: {node.traffic_level:.2f}<br>
            Last Update: {time.strftime('%H:%M:%S', time.localtime(node.last_update))}
            """
            
            # Add marker
            folium.Marker(
                location=[node.latitude, node.longitude],
                popup=popup_html,
                tooltip=node.name,
                icon=folium.Icon(color=color, icon=icon, prefix=prefix)
            ).add_to(m)
        
        # Add all edges to the map
        for edge_id, edge in self.edges.items():
            start_node = self.nodes[edge.start_node]
            end_node = self.nodes[edge.end_node]
            
            # Set color and weight based on traffic level
            color = self._get_folium_color(edge.traffic_level)
            weight = 2 + int(edge.traffic_level * 5)  # 2-7 pixels based on traffic
            
            # Create popup with edge information
            popup_html = f"""
            <b>Road Segment {edge_id}</b><br>
            Distance: {edge.distance:.2f} km<br>
            Speed Limit: {edge.speed_limit} km/h<br>
            Current Travel Time: {edge.get_travel_time():.2f} minutes<br>
            Traffic Level: {edge.traffic_level:.2f}
            """
            
            # Add polyline
            folium.PolyLine(
                locations=[[start_node.latitude, start_node.longitude], 
                          [end_node.latitude, end_node.longitude]],
                popup=popup_html,
                tooltip=f"{start_node.name} to {end_node.name}",
                color=color,
                weight=weight,
                opacity=0.8
            ).add_to(m)
        
        # If a route is specified, highlight it
        if route and len(route) > 1:
            route_points = []
            for node_id in route:
                node = self.nodes[node_id]
                route_points.append([node.latitude, node.longitude])
            
            # Add the route as a thick blue line
            folium.PolyLine(
                locations=route_points,
                color='blue',
                weight=5,
                opacity=1.0,
                tooltip='Optimal Emergency Route'
            ).add_to(m)
        
        # Save map to HTML file
        map_file = "traffic_map.html"
        m.save(map_file)
        
        # Open the map in a web browser
        try:
            webbrowser.open('file://' + os.path.abspath(map_file))
        except:
            print(f"Map generated and saved to {map_file}")
    
    def _get_folium_color(self, traffic_level):
        """Get color for Folium map based on traffic level"""
        if traffic_level < 0.3:
            return 'green'
        elif traffic_level < 0.6:
            return 'orange'
        else:
            return 'red'

# Example usage
def demo_route_optimizer():
    """Set up a demo traffic network with simulated data"""
    optimizer = RouteOptimizer()
    
    # Add nodes (traffic junctions) - Using Bengaluru (Bangalore) coordinates
    optimizer.add_node(1, "Majestic", 12.9767, 77.5713)
    optimizer.add_node(2, "Silk Board", 12.9171, 77.6223)
    optimizer.add_node(3, "Whitefield", 12.9698, 77.7500)
    optimizer.add_node(4, "Electronic City", 12.8399, 77.6770)
    optimizer.add_node(5, "MG Road", 12.9753, 77.6039)
    optimizer.add_node(6, "Hebbal", 13.0399, 77.5966)
    
    # Add edges (road segments)
    # Format: edge_id, start_node, end_node, distance (km), speed_limit (km/h)
    optimizer.add_edge(101, 1, 5, 4.5, 40)  # Majestic to MG Road
    optimizer.add_edge(102, 5, 3, 14.2, 50)  # MG Road to Whitefield
    optimizer.add_edge(103, 5, 2, 11.3, 45)  # MG Road to Silk Board
    optimizer.add_edge(104, 2, 4, 12.0, 60)  # Silk Board to Electronic City
    optimizer.add_edge(105, 1, 6, 8.8, 55)  # Majestic to Hebbal
    optimizer.add_edge(106, 6, 3, 15.5, 50)  # Hebbal to Whitefield
    optimizer.add_edge(107, 2, 3, 13.7, 40)  # Silk Board to Whitefield
    
    # Add cameras - for demo we'll use the webcam for all nodes
    for node_id in range(1, 7):
        optimizer.add_camera(node_id, 0, node_id)  # Using webcam for demo
    
    # Find an optimal route
    start_node = 1  # Majestic
    end_node = 3    # Whitefield
    route, edges = optimizer.find_optimal_route(start_node, end_node)
    
    if route:
        print(f"Optimal route found: {' -> '.join([optimizer.nodes[node_id].name for node_id in route])}")
        print(f"Estimated travel time: {optimizer.nodes[end_node].get_travel_time(0)} minutes")
    else:
        print(f"No route found from {optimizer.nodes[start_node].name} to {optimizer.nodes[end_node].name}")
    
    # Generate a map with the optimal route
    optimizer.generate_traffic_map(route)
    
    # Start monitoring (will run until interrupted)
    optimizer.start_monitoring()

if __name__ == "__main__":
    demo_route_optimizer()