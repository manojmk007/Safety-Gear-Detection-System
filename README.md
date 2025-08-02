Smart Safety Gear Detection System 🦺👷‍♂️
An AI-powered safety compliance monitoring system that uses computer vision to detect proper safety gear usage in industrial environments. Built with Python, StreamLit, and machine learning technologies.

🎯 Features
Real-time detection of safety gear including helmets, safety vests, and protective equipment
User-friendly web interface built with StreamLit
Support for both image and video processing
Detailed analytics and compliance reporting
Configurable detection parameters
🛠️ Technologies Used
Frontend: StreamLit
Backend: Python
Machine Learning Framework: TensorFlow/PyTorch
Image Processing: OpenCV
Data Analysis: NumPy, Pandas
Visualization: Matplotlib, Seaborn
Model Training: YOLO architecture
📊 Project Structure
├── SSGDS/
│   ├── runs/
│   │   └── detect/
│   │       └── train/
│   ├── stored_media/
│   │   ├── processed_videos/
│   │   └── uploaded_videos/
│   ├── app.py
│   └── Smart_Safety_Gear_Detection_System.py
🚀 Installation Guide
Clone the repository:

git clone https://github.com/yourusername/Smart-Safety-Gear-Detection-System.git
Create and activate virtual environment:

python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate  # For Windows
Install required dependencies:

pip install -r requirements.txt
Download the pre-trained weights (if applicable):

python download_weights.py
Run the application:

streamlit run app.py
📋 Usage
Access the web interface at http://localhost:8501
Upload an image or video file containing safety gear scenarios
Adjust detection parameters if needed
View real-time detection results and analytics
Export reports and statistics as needed
📈 Performance Metrics
Model Accuracy: 95%+
Real-time processing capability: 30 FPS
Support for multiple safety gear categories
Low false positive rate
🤝 Contributing
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
⚠️ Project Status and Rights
This project was developed during an internship at Infosys Limited. All rights and intellectual property associated with this project belong to Infosys Limited. This repository serves as a demonstration of the work completed during the internship period. Any use, modification, or distribution of this code should be in accordance with Infosys Limited's policies and guidelines.

🙏 Acknowledgments
Infosys Limited for providing the internship opportunity and resources
Project mentors and team members at Infosys
Dataset contributors and supporters
