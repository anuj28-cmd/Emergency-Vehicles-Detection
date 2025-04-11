# Emergency Vehicles Detection

## Project Overview
This project implements a computer vision system designed to detect emergency vehicles in traffic scenarios. The system uses deep learning models trained on Indian traffic datasets to identify emergency vehicles like ambulances and police cars, enabling priority routing and traffic management.

## Dataset
The project uses the Indian Balanced Dataset 2.v3i.yolov9, which contains:
- Images of various vehicles in Indian traffic scenarios
- Labeled data for emergency vehicles (ambulances, police vehicles)
- Training, validation, and testing splits

## Project Structure
- `data_prep/`: Scripts for dataset preparation and processing
  - `ConvertDataset.py`: Converts dataset formats
  - `Split.py`: Splits data into training and validation sets
  
- `model_training/`: Code for model development
  - `Model.py`: Implementation of the emergency vehicle detection model
  
- `main/`: Core application files
  - `Demo.py`: Demo application to run the detection system
  - `emergency_vehicle_model.h5`: Trained model weights
  - `emergency_vehicle_model_final.h5`: Final optimized model
  
- `extensions/`: Additional system components
  - `EmergencyResponseSystem.py`: Alert system for emergency vehicles
  - `RouteOptimizer.py`: Optimizes routes for detected emergency vehicles
  - `SmartTrafficSystem.py`: Traffic control integration
  
- `Test/`: Test images for model evaluation

## Installation and Setup
1. Clone this repository
2. Install required dependencies:
```
pip install tensorflow opencv-python numpy matplotlib
```
3. Download the pre-trained model or train the model yourself using the provided scripts

## Usage
Run the demo application:
```
python main/Demo.py
```

## Model Performance
The emergency vehicle detection model achieves:
- High accuracy in varying traffic conditions
- Real-time detection capability
- Robustness to different weather and lighting conditions

## Future Work
- Integration with traffic management systems
- Mobile application development
- Extended detection capabilities for additional emergency vehicle types