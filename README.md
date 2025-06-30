# Sign-Language-Detection-System
The system utilizes machine learning and computer vision techniques to predict the sign language shown by the user, either from an uploaded image or through real-time video streaming.  A special feature is included to activate the system only between 6 PM and 10 PM for demonstration purposes. This project focuses on detecting sign language gestures using **traditional machine learning methods (Random Forest, SVM)** and **computer vision feature extraction** (histograms, statistics, edges).  

# Main Goals:
- Detect a predefined set of **words** (`good`, `hello`, `no`, `thank you`, `yes`)
- Recognize **alphabet letters (A-Z)**
- Provide both **image upload** and **real-time webcam detection**
- **Allow predictions only between 6 PM - 10 PM IST (India Standard Time)**
- GUI using **Tkinter** (with Command-Line fallback)

# dataset folder structure 
- Sign-Language-Detection/
├── dataset/                          
│   ├── good/
│   │   ├── good_01.jpg
│   │   └── good_02.jpg
│   ├── hello/
│   │   ├── hello_01.jpg
│   │   └── hello_02.jpg
│   ├── no/
│   │   ├── no_01.jpg
│   │   └── no_02.jpg
│   ├── thank you/
│   │   ├── thankyou_01.jpg
│   │   └── thankyou_02.jpg
│   └── yes/
│       ├── yes_01.jpg
│       └── yes_02.jpg
│
├── asl_alphabet_train/               
│   ├── A/
│   │   ├── A_01.jpg
│   │   └── A_02.jpg
│   ├── B/
│   │   ├── B_01.jpg
│   │   └── B_02.jpg
│   ├── ...
│   └── Z/
│       ├── Z_01.jpg
│       └── Z_02.jpg
│
├── sign_language_detector.py        
├── requirements.txt                  
└── README.md                        
# How to Run
python sign_language_detector.py
