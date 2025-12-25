from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from predict import load_alzheimer_model, predict_alzheimer, get_ai_suggestions
from config import Config
import os
from datetime import datetime
import json



# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

app.secret_key = 'your-secret-key'

# Initialize database
db = SQLAlchemy(app)


# Create upload folder if doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)


# ===== DATABASE MODELS =====

class Doctor(db.Model):
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    license = db.Column(db.String(100), nullable=False, unique=True)
    specialization = db.Column(db.String(100), nullable=False)
    hospital = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    scans = db.relationship('MRIScan', backref='doctor', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'fullname': self.fullname,
            'email': self.email,
            'phone': self.phone,
            'specialization': self.specialization,
            'hospital': self.hospital,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    medical_history = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    scans = db.relationship('MRIScan', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'fullname': self.fullname,
            'email': self.email,
            'phone': self.phone,
            'age': self.age,
            'gender': self.gender,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class MRIScan(db.Model):
    __tablename__ = 'mri_scans'
    
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(100), default='Pending')
    confidence = db.Column(db.Float, default=0.0)
    stage = db.Column(db.String(50), default='')
    ai_suggestions = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'doctor_id': self.doctor_id,
            'patient_id': self.patient_id,
            'filename': self.filename,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'stage': self.stage,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


# ===== HELPER FUNCTIONS =====

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ===== SERVE HTML PAGES =====
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/index.html')
def index_html():
    return send_from_directory('.', 'index.html')


@app.route('/login.html')
def login_html():
    return send_from_directory('.', 'login.html')


@app.route('/doctor_register.html')
def doctor_register_html():
    return send_from_directory('.', 'doctor_register.html')


@app.route('/patient_register.html')
def patient_register_html():
    return send_from_directory('.', 'patient_register.html')


@app.route('/doctor_dashboard.html')
def doctor_dashboard_html():
    return send_from_directory('.', 'doctor_dashboard.html')


@app.route('/patient_dashboard.html')
def patient_dashboard_html():
    return send_from_directory('.', 'patient_dashboard.html')


@app.route('/static/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('static/css', filename)


@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename)


# ===== DOCTOR REGISTRATION =====
@app.route('/api/doctor-register', methods=['POST'])
def doctor_register():
    try:
        data = request.get_json()
        
        # Validate
        if not all([data.get('fullname'), data.get('email'), data.get('password'), 
                   data.get('phone'), data.get('license'), data.get('specialization'), 
                   data.get('hospital')]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
        
        # Check if email exists
        if Doctor.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        # Check if license exists
        if Doctor.query.filter_by(license=data['license']).first():
            return jsonify({'success': False, 'message': 'License already registered'}), 400
        
        # Create doctor
        doctor = Doctor(
            fullname=data['fullname'],
            email=data['email'],
            phone=data['phone'],
            license=data['license'],
            specialization=data['specialization'],
            hospital=data['hospital']
        )
        doctor.set_password(data['password'])
        
        db.session.add(doctor)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Doctor registered successfully'}), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== PATIENT REGISTRATION =====
@app.route('/api/patient-register', methods=['POST'])
def patient_register():
    try:
        data = request.get_json()
        
        # Validate
        if not all([data.get('fullname'), data.get('email'), data.get('password'), 
                   data.get('phone'), data.get('age'), data.get('gender')]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
        
        # Check if email exists
        if Patient.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        # Create patient
        patient = Patient(
            fullname=data['fullname'],
            email=data['email'],
            phone=data['phone'],
            age=int(data['age']),
            gender=data['gender'],
            medical_history=data.get('medical_history', '')
        )
        patient.set_password(data['password'])
        
        db.session.add(patient)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Patient registered successfully'}), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== DOCTOR LOGIN =====
@app.route('/api/doctor-login', methods=['POST'])
def doctor_login():
    try:
        data = request.get_json()
        
        # Validate
        if not data.get('email') or not data.get('password'):
            return jsonify({'success': False, 'message': 'Email and password required'}), 400
        
        # Find doctor
        doctor = Doctor.query.filter_by(email=data['email']).first()
        
        if not doctor or not doctor.check_password(data['password']):
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
        
        # Set session
        session['doctor_id'] = doctor.id
        session['doctor_name'] = doctor.fullname
        session.permanent = True
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'doctor': doctor.to_dict()
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== PATIENT LOGIN =====
@app.route('/api/patient-login', methods=['POST'])
def patient_login():
    try:
        data = request.get_json()
        
        # Validate
        if not data.get('email') or not data.get('password'):
            return jsonify({'success': False, 'message': 'Email and password required'}), 400
        
        # Find patient
        patient = Patient.query.filter_by(email=data['email']).first()
        
        if not patient or not patient.check_password(data['password']):
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
        
        # Set session
        session['patient_id'] = patient.id
        session['patient_name'] = patient.fullname
        session.permanent = True
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'patient': patient.to_dict()
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== LOGOUT =====
@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200


# ===== DOCTOR DASHBOARD =====
@app.route('/api/doctor-dashboard', methods=['GET'])
def doctor_dashboard():
    if 'doctor_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        doctor = Doctor.query.get(session['doctor_id'])
        scans = MRIScan.query.filter_by(doctor_id=doctor.id).all()
        
        return jsonify({
            'success': True,
            'doctor': doctor.to_dict(),
            'scans': [scan.to_dict() for scan in scans]
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== MRI UPLOAD & PREDICTION =====
@app.route('/api/predict-mri', methods=['POST'])
def predict_mri():
    print("=== /api/predict-mri ===")
    print("Files:", request.files)
    print("Form:", request.form)
    print("Keys in request.files:", list(request.files.keys()))
    print("Keys in request.form:", list(request.form.keys()))
    if 'doctor_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        # Check file
        if 'mri_file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['mri_file']
        patient_email = request.form.get('patient_email')
        
        if not patient_email:
            return jsonify({'success': False, 'message': 'Patient email required'}), 400
        
        # Find patient
        patient = Patient.query.filter_by(email=patient_email).first()
        if not patient:
            return jsonify({'success': False, 'message': 'Patient not found'}), 400
        
        # Save file
        filename = secure_filename(file.filename)  # Original extension keep chestundi
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        result = predict_alzheimer(filepath)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Get suggestions
        suggestions = get_ai_suggestions(result['prediction'])
        
        # Save to database
        scan = MRIScan(
            doctor_id=session['doctor_id'],
            patient_id=patient.id,
            filename=filename,
            filepath=filepath,
            prediction=result['prediction'],
            confidence=result['confidence'],
            ai_suggestions=suggestions
        )
        db.session.add(scan)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'] * 100, 2),
            'classes': result['classes'],
            'suggestions': suggestions,
            'scan_id': scan.id
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== PATIENT DASHBOARD =====
@app.route('/api/patient-dashboard', methods=['GET'])
def patient_dashboard():
    if 'patient_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        patient = Patient.query.get(session['patient_id'])
        scans = MRIScan.query.filter_by(patient_id=patient.id).all()
        
        return jsonify({
            'success': True,
            'patient': patient.to_dict(),
            'scans': [scan.to_dict() for scan in scans]
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ===== CHECK SESSION =====
@app.route('/api/check-session', methods=['GET'])
def check_session():
    return jsonify({
        'doctor_logged_in': 'doctor_id' in session,
        'patient_logged_in': 'patient_id' in session,
        'doctor_name': session.get('doctor_name'),
        'patient_name': session.get('patient_name')
    }), 200


# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'success': False, 'message': 'Server error'}), 500


# ===== CREATE DATABASE =====
def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully!")


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
