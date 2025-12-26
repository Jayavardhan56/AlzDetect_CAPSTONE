🧠 AI‑Assisted Early Alzheimer’s Prediction

An AI‑powered web application that helps doctors analyze brain MRI scans and predict the stage of Alzheimer’s disease using a deep learning model built on EfficientNetB3.

🌟 Project Overview

🩺 Goal: Support early detection of Alzheimer’s by classifying brain MRI scans into four stages.

👨‍⚕️ Users: Doctors (upload & analyze MRIs, view reports) and Patients (view their results shared by doctors).

🤖 Core AI: Transfer‑learning model (EfficientNetB3) trained on 4 MRI classes, achieving high validation accuracy on the dataset.

🌐 App Type: Full‑stack Flask web app with secure authentication, dashboards, and MRI upload & prediction workflow.

🧬 Alzheimer’s Classes

The model predicts one of the following classes:

🟢 Non‑Demented

🟡 Very Mild Demented

🟠 Mild Demented

🔴 Moderate Demented

Each prediction is accompanied by class probabilities and AI‑generated suggestions for next clinical steps.

🛠️ Tech Stack

Backend & ML

🐍 Python (3.x)

🌶️ Flask – REST API & web server

🧠 TensorFlow / Keras – EfficientNetB3 image classifier

📦 NumPy, Pillow – image preprocessing and array ops

🗄️ SQLAlchemy + SQLite – persistent storage for users, scans, and reports

Frontend

📄 HTML5, 🎨 CSS3

⚡ JavaScript (vanilla) – form handling & API calls

📊 Responsive dashboard UI for doctors and patients

Dev & Tools

🧪 Kaggle – model training & experimentation

💻 VS Code – development

🐙 Git & GitHub – version control

🔍 Key Features 📤 MRI Upload & Auto‑Preprocessing

Accepts raw brain MRI images of various sizes.

Automatically center‑crops, resizes to 224 × 224 224×224, converts to RGB, and applies the same preprocessing used during training.

🧠 Alzheimer’s Stage Prediction

EfficientNetB3 model outputs a probability distribution over the four dementia stages.

Returns predicted class and per‑class probabilities for transparency.

🚫 Non‑Brain Image Protection

Lightweight heuristic check to reject images that do not look like brain MRIs and ask the user to re‑upload a valid scan.

👨‍⚕️ Doctor Portal

Secure login & session management.

Dashboard showing patients, uploaded scans, and AI predictions.

👤 Patient View (Optional)

Patients can view their prediction reports shared by the doctor (depending on configuration).

💡 AI Health Suggestions

Text suggestions tailored to the predicted class to support clinical decision‑making (not a substitute for diagnosis).

🏗️ Project Structure (High Level):

alzheimer-prediction-app/ 

├── app.py # Flask app & API routes 

├── predict.py # Model loading & MRI prediction logic 

├── alz_effnet_clean.keras # Trained EfficientNetB3 model 

├── templates/ # HTML templates (login, dashboards, etc.) 

├── static/ │ 

├── css/ # Stylesheets 

│ └── js/ # Frontend scripts 

├── uploads/ # Stored MRI scans (runtime) 

├── database/ # SQLite DB file

└── requirements.txt # Python dependencies

🚀 Getting Started

🧰 Clone the repo

git clone https://github.com/your-username/alzheimer-prediction-app.git cd alzheimer-prediction-app

🐍 Create & activate virtual environment

python -m venv .venv ..venv\Scripts\activate # Windows

source .venv/bin/activate # Linux/macOS
📦 Install dependencies

pip install -r requirements.txt

🧠 Place the model

Ensure alz_effnet_clean.keras is in the project root (same folder as app.py and predict.py).

▶️ Run the app

python app.py Open http://127.0.0.1:5000 in your browser.

⚠️ Disclaimer This application is built for research and educational purposes only.

It is not a certified medical device and must not be used as a standalone diagnostic tool.

Clinical decisions must always be made by qualified healthcare professionals.
