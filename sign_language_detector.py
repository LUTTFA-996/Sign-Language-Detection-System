# Simple Sign Language Detection System
# Uses scikit-learn instead of TensorFlow for easier installation

import os
import cv2 # type: ignore
import numpy as np
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore
import threading
import time

# Try to import tkinter, fallback to command line if not available
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    print("Tkinter not available. Using command line mode.")
    TKINTER_AVAILABLE = False

class SimpleSignLanguageDetector:
    def __init__(self):
        # Setup GUI or command line
        if TKINTER_AVAILABLE:
            self.gui_mode = True
            self.root = tk.Tk()
            self.root.title("Sign Language Detection System")
            self.root.geometry("800x600")
            # Added a protocol for window close event to ensure proper cleanup
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        else:
            self.gui_mode = False
            self.root = None

        # Paths - Update these to your actual paths
        self.base_path = r"C:\Users\Lutifah\Desktop\INTERSHIP\Sign-Language"
        self.words_path = os.path.join(self.base_path, "dataset")
        self.letters_path = os.path.join(self.base_path, "asl_alphabet_train")

        # Model variables
        self.word_model = None
        self.letter_model = None
        self.word_label_encoder = LabelEncoder()
        self.letter_label_encoder = LabelEncoder()
        self.word_classes = ['good', 'hello', 'no', 'thank you', 'yes']
        self.letter_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        # Video capture
        self.cap = None
        self.video_running = False

        # Time restriction
        self.time_restricted = True
        self.start_hour = 18  # 6 PM IST
        self.end_hour = 22    # 10 PM IST

        if self.gui_mode:
            self.setup_gui()
        else:
            self.setup_command_line()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        for i in range(7): # Configure rows to expand
            main_frame.rowconfigure(i, weight=1)
        for i in range(3): # Configure columns to expand
            main_frame.columnconfigure(i, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Sign Language Detection System",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="n")

        # Time status
        self.time_status_label = ttk.Label(main_frame, text="", font=("Arial", 10))
        self.time_status_label.grid(row=1, column=0, columnspan=3, pady=5, sticky="n")
        self.update_time_status()

        # Model training section
        train_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="10")
        train_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        train_frame.columnconfigure(0, weight=1)
        train_frame.columnconfigure(1, weight=1)
        train_frame.columnconfigure(2, weight=1)

        ttk.Button(train_frame, text="Train Word Model",
                   command=self.train_word_model).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(train_frame, text="Train Letter Model",
                   command=self.train_letter_model).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(train_frame, text="Load Models",
                   command=self.load_models).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Image upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Image Upload Detection", padding="10")
        upload_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        upload_frame.columnconfigure(0, weight=1)
        upload_frame.columnconfigure(1, weight=1)

        ttk.Button(upload_frame, text="Upload Image for Word Detection",
                   command=self.upload_word_image).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(upload_frame, text="Upload Image for Letter Detection",
                   command=self.upload_letter_image).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Video detection section
        video_frame = ttk.LabelFrame(main_frame, text="Real-time Video Detection", padding="10")
        video_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        video_frame.columnconfigure(0, weight=1)
        video_frame.columnconfigure(1, weight=1)
        video_frame.columnconfigure(2, weight=1)

        ttk.Button(video_frame, text="Start Word Detection",
                   command=self.start_word_video).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(video_frame, text="Start Letter Detection",
                   command=self.start_letter_video).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(video_frame, text="Stop Video",
                   command=self.stop_video).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Results display
        self.result_label = ttk.Label(main_frame, text="Results will appear here",
                                       font=("Arial", 12), background="lightgray", anchor="center") # Centered text
        self.result_label.grid(row=5, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

    def setup_command_line(self):
        print("Sign Language Detection System - Command Line Mode")
        print("=" * 60)
        self.print_time_status()

    def update_time_status(self):
        current_time = datetime.now()
        current_hour = current_time.hour

        # Current time in Thiruvananthapuram is Monday, June 30, 2025 at 12:52:10 AM IST.
        # This means the current hour is 0 (midnight).
        # Your system is active from 6 PM (18) to 10 PM (22).
        # So, the system will be inactive at this exact moment.

        if self.time_restricted:
            if self.start_hour <= current_hour < self.end_hour:
                status = f" System Active (Current time: {current_time.strftime('%H:%M')})"
                if self.gui_mode:
                    self.time_status_label.config(foreground="green")
            else:
                status = f" System Inactive - Available from {self.start_hour}:00 to {self.end_hour}:00 (Current: {current_time.strftime('%H:%M')})"
                if self.gui_mode:
                    self.time_status_label.config(foreground="red")
        else:
            status = f" System Active (Time restriction disabled)"
            if self.gui_mode:
                self.time_status_label.config(foreground="green")

        if self.gui_mode:
            self.time_status_label.config(text=status)
            self.root.after(60000, self.update_time_status) # Update every minute

    def print_time_status(self):
        current_time = datetime.now()
        current_hour = current_time.hour

        if self.time_restricted:
            if self.start_hour <= current_hour < self.end_hour:
                print(f" System Active (Current time: {current_time.strftime('%H:%M')})")
            else:
                print(f" System Inactive - Available from {self.start_hour}:00 to {self.end_hour}:00 (Current: {current_time.strftime('%H:%M')})")
        else:
            print(f" System Active (Time restriction disabled)")

    def is_system_active(self):
        if not self.time_restricted:
            return True
        current_hour = datetime.now().hour
        return self.start_hour <= current_hour < self.end_hour

    def extract_features(self, image_path, target_size=(64, 64)):
        """Extract features from image using traditional computer vision methods"""
        try:
            # Read and resize image
            img = cv2.imread(image_path)
            if img is None:
                # print(f"Warning: Could not read image at {image_path}") # For debugging feature extraction issues
                return None

            img = cv2.resize(img, target_size)

            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Feature extraction
            features = []

            # 1. Histogram features
            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])

            features.extend(hist_gray.flatten())
            features.extend(hist_h.flatten())
            features.extend(hist_s.flatten())

            # 2. Statistical features
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),
                np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),
                np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
            ])

            # 3. Edge features
            edges = cv2.Canny(gray, 50, 150)
            features.extend([
                np.sum(edges), np.mean(edges), np.std(edges)
            ])

            return np.array(features)

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def load_dataset(self, data_path, classes):
        """Load dataset and extract features"""
        features = []
        labels = []

        print(f"Loading dataset from: {data_path}")

        for class_name in classes:
            class_path = os.path.join(data_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path '{class_path}' does not exist for class '{class_name}'. Skipping.")
                continue

            class_features_count = 0
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, filename)
                    feature_vector = self.extract_features(img_path)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(class_name)
                        class_features_count += 1

            print(f"Loaded {class_features_count} images for class '{class_name}' from '{class_path}'")

        if len(features) == 0:
            raise ValueError("No images found in the dataset for training.")

        return np.array(features), np.array(labels)

    def train_word_model(self):
        """Train word recognition model"""
        if not self.is_system_active():
            if self.gui_mode:
                messagebox.showwarning("System Inactive", "System is only active from 6 PM to 10 PM IST.")
            else:
                print("System is only active from 6 PM to 10 PM IST.")
            return

        def train():
            try:
                if self.gui_mode:
                    self.progress.start()
                    self.result_label.config(text="Training word model...")
                else:
                    print("Training word model...")

                # Load dataset
                X, y = self.load_dataset(self.words_path, self.word_classes)

                # Encode labels
                y_encoded = self.word_label_encoder.fit_transform(y)

                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )

                # Train model
                print("Training Random Forest model for words...")
                self.word_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                )
                self.word_model.fit(X_train, y_train)

                # Evaluate
                y_pred = self.word_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Save model
                with open('word_model.pkl', 'wb') as f:
                    pickle.dump((self.word_model, self.word_label_encoder), f)

                result_text = f"Word model trained! Accuracy: {accuracy:.2f}"
                print(result_text)
                print("\nClassification Report (Word Model):")
                print(classification_report(y_test, y_pred,
                                            target_names=self.word_label_encoder.classes_))

                if self.gui_mode:
                    self.progress.stop()
                    self.result_label.config(text=result_text)

            except ValueError as ve:
                error_text = f"Dataset Error: {str(ve)}. Ensure 'dataset' folder contains subfolders for 'good', 'hello', 'no', 'thank you', 'yes' with images inside."
                print(error_text)
                if self.gui_mode:
                    self.progress.stop()
                    messagebox.showerror("Dataset Error", error_text)
                    self.result_label.config(text=error_text)
            except Exception as e:
                error_text = f"Error training word model: {str(e)}"
                print(error_text)
                if self.gui_mode:
                    self.progress.stop()
                    messagebox.showerror("Training Error", error_text)
                    self.result_label.config(text=error_text)

        if self.gui_mode:
            threading.Thread(target=train, daemon=True).start()
        else:
            train()

    def train_letter_model(self):
        """Train letter recognition model"""
        if not self.is_system_active():
            if self.gui_mode:
                messagebox.showwarning("System Inactive", "System is only active from 6 PM to 10 PM IST.")
            else:
                print("System is only active from 6 PM to 10 PM IST.")
            return

        def train():
            try:
                if self.gui_mode:
                    self.progress.start()
                    self.result_label.config(text="Training letter model...")
                else:
                    print("Training letter model...")

                # Load dataset
                X, y = self.load_dataset(self.letters_path, self.letter_classes)

                # Encode labels
                y_encoded = self.letter_label_encoder.fit_transform(y)

                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )

                # Train model
                print("Training SVM model for letters...")
                self.letter_model = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
                self.letter_model.fit(X_train, y_train)

                # Evaluate
                y_pred = self.letter_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Save model
                with open('letter_model.pkl', 'wb') as f:
                    pickle.dump((self.letter_model, self.letter_label_encoder), f)

                result_text = f"Letter model trained! Accuracy: {accuracy:.2f}"
                print(result_text)
                print("\nClassification Report (Letter Model):")
                print(classification_report(y_test, y_pred,
                                            target_names=self.letter_label_encoder.classes_))

                if self.gui_mode:
                    self.progress.stop()
                    self.result_label.config(text=result_text)

            except ValueError as ve:
                error_text = f"Dataset Error: {str(ve)}. Ensure 'asl_alphabet_train' folder contains subfolders for A-Z with images inside."
                print(error_text)
                if self.gui_mode:
                    self.progress.stop()
                    messagebox.showerror("Dataset Error", error_text)
                    self.result_label.config(text=error_text)
            except Exception as e:
                error_text = f"Error training letter model: {str(e)}"
                print(error_text)
                if self.gui_mode:
                    self.progress.stop()
                    messagebox.showerror("Training Error", error_text)
                    self.result_label.config(text=error_text)

        if self.gui_mode:
            threading.Thread(target=train, daemon=True).start()
        else:
            train()

    def load_models(self):
        """Load saved models"""
        try:
            status = []

            if os.path.exists('word_model.pkl'):
                with open('word_model.pkl', 'rb') as f:
                    self.word_model, self.word_label_encoder = pickle.load(f)
                status.append("Word model loaded")
            else:
                status.append("Word model PKL not found")

            if os.path.exists('letter_model.pkl'):
                with open('letter_model.pkl', 'rb') as f:
                    self.letter_model, self.letter_label_encoder = pickle.load(f)
                status.append("Letter model loaded")
            else:
                status.append("Letter model PKL not found")

            result_text = " | ".join(status) if status else "No models found. Please train first."
            print(result_text)

            if self.gui_mode:
                self.result_label.config(text=result_text)

        except Exception as e:
            error_text = f"Error loading models: {str(e)}"
            print(error_text)
            if self.gui_mode:
                messagebox.showerror("Load Error", error_text)
                self.result_label.config(text=error_text)

    def predict_image(self, image_path, model_type):
        """Predict image using trained model"""
        if model_type == 'word':
            model = self.word_model
            label_encoder = self.word_label_encoder
        else:
            model = self.letter_model
            label_encoder = self.letter_label_encoder

        if model is None:
            # This case should ideally be caught by the calling function (upload_word/letter_image)
            return None, 0

        features = self.extract_features(image_path)
        if features is None:
            print(f"Warning: No features extracted from {image_path}") # Debugging for bad images
            return None, 0

        # Predict
        prediction_encoded = model.predict([features])[0]
        confidence = np.max(model.predict_proba([features]))

        # Decode label
        predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]

        return predicted_label, confidence

    def upload_word_image(self):
        """Upload and predict word from image"""
        if not self.is_system_active():
            messagebox.showwarning("System Inactive", "System is only active from 6 PM to 10 PM IST.")
            return

        if self.word_model is None:
            messagebox.showwarning("Model Error", "Please train or load the word model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select image for word detection",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            predicted_word, confidence = self.predict_image(file_path, 'word')
            if predicted_word:
                result = f"Predicted Word: {predicted_word} (Confidence: {confidence:.2f})"
                self.result_label.config(text=result)
            else:
                self.result_label.config(text="Error: Could not process image for word detection")

    def upload_letter_image(self):
        """Upload and predict letter from image"""
        if not self.is_system_active():
            messagebox.showwarning("System Inactive", "System is only active from 6 PM to 10 PM IST.")
            return

        if self.letter_model is None:
            messagebox.showwarning("Model Error", "Please train or load the letter model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select image for letter detection",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            predicted_letter, confidence = self.predict_image(file_path, 'letter')
            if predicted_letter:
                result = f"Predicted Letter: {predicted_letter} (Confidence: {confidence:.2f})"
                self.result_label.config(text=result)
            else:
                self.result_label.config(text="Error: Could not process image for letter detection")

    def start_word_video(self):
        """Start real-time word detection"""
        if not self.is_system_active():
            if self.gui_mode:
                messagebox.showwarning("System Inactive", "System is only active from 6 PM to 10 PM IST.")
            else:
                print("System is only active from 6 PM to 10 PM IST.")
            return

        if self.word_model is None:
            if self.gui_mode:
                messagebox.showwarning("Model Error", "Please train or load the word model first")
            else:
                print("Please train or load the word model first")
            return

        self.start_video_detection('word')

    def start_letter_video(self):
        """Start real-time letter detection"""
        if not self.is_system_active():
            if self.gui_mode:
                messagebox.showwarning("System Inactive", "System is only active from 6 PM to 10 PM IST.")
            else:
                print("System is only active from 6 PM to 10 PM IST.")
            return

        if self.letter_model is None:
            if self.gui_mode:
                messagebox.showwarning("Model Error", "Please train or load the letter model first")
            else:
                print("Please train or load the letter model first")
            return

        self.start_video_detection('letter')

    def start_video_detection(self, detection_type):
        """Start video detection"""
        if self.video_running:
            if self.gui_mode:
                messagebox.showwarning("Video Active", "Video detection is already running. Please stop it first.")
            else:
                print("Video detection is already running. Please stop it first.")
            return

        self.cap = cv2.VideoCapture(0) # 0 for default camera
        if not self.cap.isOpened():
            if self.gui_mode:
                messagebox.showerror("Camera Error", "Could not open camera. Make sure it's connected and not in use by another application.")
            else:
                print("Could not open camera. Make sure it's connected and not in use by another application.")
            self.cap = None # Ensure cap is None if it failed to open
            return

        self.video_running = True

        def detect():
            temp_image_path = "temp_frame.jpg"
            last_prediction_time = time.time()
            prediction_interval = 0.5 # Predict every 0.5 seconds to reduce load

            while self.video_running and self.is_system_active():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                    break

                # Create ROI
                h, w, _ = frame.shape
                roi_size = 200
                x1 = w // 2 - roi_size // 2
                y1 = h // 2 - roi_size // 2
                x2 = x1 + roi_size
                y2 = y1 + roi_size

                # Ensure ROI coordinates are within bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Draw ROI rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Place hand here", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                current_prediction_label = "N/A" # Default label
                current_prediction_confidence = 0.0

                # Only predict if enough time has passed
                if time.time() - last_prediction_time > prediction_interval:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        try:
                            # Save ROI to a temporary file
                            cv2.imwrite(temp_image_path, roi)
                            predicted_label, confidence = self.predict_image(temp_image_path, detection_type)

                            if predicted_label: # and confidence > 0.6: # Optional: only show if confidence is high
                                current_prediction_label = predicted_label
                                current_prediction_confidence = confidence
                                if self.gui_mode:
                                    self.result_label.config(text=f"Live {detection_type.title()}: {predicted_label} (Conf: {confidence:.2f})")
                                else:
                                    print(f"\rLive {detection_type.title()}: {predicted_label} (Conf: {confidence:.2f})", end="", flush=True)

                            # Clean up temporary image immediately after use
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                        except Exception as e:
                            print(f"Error during live prediction: {e}")
                    last_prediction_time = time.time() # Update last prediction time

                # Display the current prediction on the OpenCV frame
                display_text = f"Predicted: {current_prediction_label} (Conf: {current_prediction_confidence:.2f})"
                cv2.putText(frame, display_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                cv2.imshow(f'Sign Language Detection - {detection_type.title()}', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else: # This else block runs if the while loop finishes normally (e.g., system inactive)
                if not self.is_system_active():
                    message = "System became inactive due to time restriction."
                    if self.gui_mode:
                        messagebox.showwarning("System Inactive", message)
                    else:
                        print(f"\n{message}")

            # Cleanup when the loop breaks
            self.stop_video_internal() # Call an internal helper to avoid re-triggering messages

        threading.Thread(target=detect, daemon=True).start()

    def stop_video_internal(self):
        """Internal method to stop video without extra messages/popups."""
        self.video_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None # Reset cap to None
        cv2.destroyAllWindows()

    def stop_video(self):
        """Stop video detection and update GUI/CLI."""
        self.stop_video_internal() # Call the internal helper
        result_text = "Video detection stopped"
        print(result_text)
        if self.gui_mode:
            self.result_label.config(text=result_text)
            self.root.update_idletasks() # Ensure GUI updates immediately

    def on_closing(self):
        """Handle window closing event."""
        self.stop_video_internal()
        if self.gui_mode:
            self.root.destroy()

    def run_command_line(self):
        """Run the command line interface"""
        while True:
            print("\n" + "="*50)
            print("Sign Language Detection System")
            print("="*50)
            self.print_time_status()
            print("\nOptions:")
            print("1. Train Word Model")
            print("2. Train Letter Model")
            print("3. Load Models")
            print("4. Test Image (Word)")
            print("5. Test Image (Letter)")
            print("6. Start Video Detection (Word)")
            print("7. Start Video Detection (Letter)")
            print("8. Toggle Time Restriction")
            print("9. Check Dataset Status")
            print("10. Exit")

            try:
                choice = input("\nEnter your choice (1-10): ").strip()

                if choice == '1':
                    self.train_word_model()
                elif choice == '2':
                    self.train_letter_model()
                elif choice == '3':
                    self.load_models()
                elif choice == '4':
                    self.test_image_cli('word')
                elif choice == '5':
                    self.test_image_cli('letter')
                elif choice == '6':
                    self.start_word_video()
                elif choice == '7':
                    self.start_letter_video()
                elif choice == '8':
                    self.time_restricted = not self.time_restricted
                    print(f"Time restriction {'enabled' if self.time_restricted else 'disabled'}")
                elif choice == '9':
                    prepare_datasets() # Call the standalone function
                elif choice == '10':
                    print("Exiting...")
                    break
                else:
                    print("Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    def test_image_cli(self, detection_type):
        """Test image in CLI mode"""
        if not self.is_system_active():
            print("System is only active from 6 PM to 10 PM IST.")
            return

        model = self.word_model if detection_type == 'word' else self.letter_model

        if model is None:
            print(f"Please train or load the {detection_type} model first")
            return

        image_path = input(f"Enter path to image for {detection_type} detection: ").strip()

        if not os.path.exists(image_path):
            print("Image file not found!")
            return

        predicted_label, confidence = self.predict_image(image_path, detection_type)
        if predicted_label:
            print(f"Predicted {detection_type.title()}: {predicted_label} (Confidence: {confidence:.2f})")
        else:
            print("Error: Could not process image")

    def run(self):
        """Run the application"""
        try:
            if self.gui_mode:
                self.root.mainloop()
            else:
                self.run_command_line()
        finally:
            # Ensure video is stopped and windows are closed even if mainloop exits unexpectedly
            self.stop_video_internal()

# Dataset preparation utility (kept as a standalone function for CLI option 9)
def prepare_datasets():
    """Check dataset status"""
    print("Dataset Preparation Helper")
    print("=" * 40)

    base_path = r"C:\Users\Lutifah\Desktop\INTERSHIP\Sign-Language"

    # Check word dataset
    words_path = os.path.join(base_path, "dataset")
    word_classes = ['good', 'hello', 'no', 'thank you', 'yes']

    print("Word Dataset Status:")
    for word in word_classes:
        word_path = os.path.join(words_path, word)
        if os.path.exists(word_path):
            count = len([f for f in os.listdir(word_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {word}: {count} images")
        else:
            print(f"   {word}: Folder '{word_path}' not found")

    # Check letter dataset
    letters_path = os.path.join(base_path, "asl_alphabet_train")
    letter_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    print("\nLetter Dataset Status:")
    if os.path.exists(letters_path):
        for letter in letter_classes:
            letter_path = os.path.join(letters_path, letter)
            if os.path.exists(letter_path):
                count = len([f for f in os.listdir(letter_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"   {letter}: {count} images")
            else:
                print(f"   {letter}: Folder '{letter_path}' not found")
    else:
        print(f"   Letters dataset base path '{letters_path}' not found")

    print("\nDataset Directory Structure:")
    print("Expected structure in 'C:\\Users\\Lutifah\\Desktop\\INTERSHIP\\Sign-Language\\':")
    print(" ├── dataset/")
    print(" │    ├── good/")
    print(" │    ├── hello/")
    print(" │    ├── no/")
    print(" │    ├── thank you/")
    print(" │    └── yes/")
    print(" └── asl_alphabet_train/")
    print("      ├── A/")
    print("      ├── B/")
    print("      └── ... (up to Z, also check for J and Z as they are often handled specially in ASL datasets)")


def create_sample_dataset(): # This function is useful for initial setup, but not directly used by the main app logic after datasets are there
    """Create sample dataset structure for demonstration."""
    print("Creating sample dataset structure...")

    base_path = "sample_data" # Using a local folder for sample data
    words_path = os.path.join(base_path, "dataset")
    letters_path = os.path.join(base_path,  "asl_alphabet_train")

    word_classes = ['good', 'hello', 'no', 'thank you', 'yes']
    letter_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    # Create word directories
    for word in word_classes:
        path = os.path.join(words_path, word)
        os.makedirs(path, exist_ok=True)
        # For actual use, you'd populate these with images.
        # Example: Image.new('RGB', (64, 64), color = 'red').save(os.path.join(path, f"{word}_01.png"))
        print(f"Created: {path}")

    # Create letter directories
    for letter in letter_classes:
        path = os.path.join(letters_path, letter)
        os.makedirs(path, exist_ok=True)
        # Example: Image.new('RGB', (64, 64), color = 'blue').save(os.path.join(path, f"{letter}_01.png"))
        print(f"Created: {path}")

    print("Sample dataset structure created. You need to add actual images to these folders for training.")


# --- THIS IS THE CRUCIAL PART ---
# This block ensures the application runs when the script is executed directly.
if __name__ == "__main__":
    app = SimpleSignLanguageDetector()
    app.run()