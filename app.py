from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from datetime import datetime
import qrcode
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app connection
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
IMG_SIZE = 224
TFLITE_MODEL_PATH = "pure_model.tflite"
class_names = sorted(['nv', 'mel', 'bkl', 'bcc', 'akiec', 'normal', 'vasc', 'df'])

# Disease information database
disease_info = {
    'akiec': {
        'name': 'Actinic keratoses',
        'description': 'Rough, scaly patches on the skin caused by years of sun exposure. Can develop into skin cancer if left untreated.',
        'suggestion': 'Consult a dermatologist. Treatment options include freezing, topical medications, or laser therapy.',
        'cure_days': '2-4 weeks with proper treatment',
    },
    'bcc': {
        'name': 'Basal cell carcinoma',
        'description': 'Most common type of skin cancer. Slow-growing and rarely spreads to other parts of the body.',
        'suggestion': 'Immediate dermatologist consultation required. Surgical removal is typically recommended.',
        'cure_days': '30-45 days after surgery or therapy',
    },
    'bkl': {
        'name': 'Benign keratosis-like lesions',
        'description': 'Non-cancerous skin growths that appear as brown, black, or tan spots. Usually harmless.',
        'suggestion': 'Monitor for changes. Visit dermatologist if it grows, bleeds, or changes color.',
        'cure_days': 'No treatment needed unless cosmetic concern',
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Common benign skin growth. Appears as a small, firm bump on the skin, usually on legs.',
        'suggestion': 'Usually harmless. Consult dermatologist only if it becomes painful or changes.',
        'cure_days': 'No treatment required; surgical removal if desired',
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'A serious type of skin cancer that develops from pigment-producing cells. Can spread to other organs if not treated early.',
        'suggestion': 'URGENT: Consult oncologist/dermatologist immediately. Early detection is crucial for successful treatment.',
        'cure_days': '30-90 days depending on stage and treatment plan',
    },
    'nv': {
        'name': 'Melanocytic nevi',
        'description': 'Common moles - usually benign growths. Most people have 10-40 moles on their body.',
        'suggestion': 'Monitor for ABCDE signs (Asymmetry, Border, Color, Diameter, Evolution). Check every 3 months.',
        'cure_days': 'Usually harmless; no treatment needed',
    },
    'vasc': {
        'name': 'Vascular lesions',
        'description': 'Birthmarks or growths made up of blood vessels. Usually harmless but can bleed if injured.',
        'suggestion': 'Consult dermatologist if it bleeds, grows rapidly, or causes discomfort.',
        'cure_days': '15-30 days with laser treatment if needed',
    },
    'normal': {
        'name': 'Normal',
        'description': 'No skin disease detected. Your skin appears healthy with no visible abnormalities.',
        'suggestion': 'Continue regular skin care routine. Protect skin from excessive sun exposure and monitor for any changes.',
        'cure_days': 'No treatment needed',
    },
    'unknown': {
        'name': 'Unknown Disease',
        'description': 'The AI model could not confidently identify the skin condition. This could be due to image quality, lighting, or an uncommon condition.',
        'suggestion': 'RECOMMENDED: Please consult a dermatologist for proper diagnosis. The image may need better lighting or a clearer view of the affected area.',
        'cure_days': 'Professional diagnosis required',
    },
}

# Load model
interpreter = None

def load_model():
    global interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image):
    """Convert PIL Image to preprocessed numpy array"""
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_resized, axis=0).astype(np.float32)
    )
    return img_array

def predict_image(image):
    """Run prediction on image"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img_array = preprocess_image(image)
    img_array = img_array.astype(input_details[0]['dtype'])
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    preds = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(preds[0][pred_idx])
    pred_label = class_names[pred_idx]
    
    if confidence < 0.60:
        pred_label = 'unknown'
    
    return pred_label, confidence

def generate_qr_data(disease_key, confidence):
    """Generate text data for QR code"""
    info = disease_info.get(disease_key, {})
    disease_name = info.get('name', 'Unknown')
    date = datetime.now().strftime('%Y-%m-%d')
    
    qr_data = f"""=== SKIN DISEASE DETECTION REPORT ===

Date: {date}

DETECTED CONDITION:
{disease_name}

CONFIDENCE: {confidence*100:.1f}%

DESCRIPTION:
{info.get('description', 'N/A')}

MEDICAL ADVICE:
{info.get('suggestion', 'N/A')}

TREATMENT DURATION:
{info.get('cure_days', 'N/A')}

---
Note: This is an AI-generated report. Please consult a dermatologist for professional medical advice."""
    
    return qr_data

def create_qr_code(data):
    """Create QR code image from data"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def generate_pdf(disease_key, confidence):
    """Generate PDF report with QR code"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    info = disease_info.get(disease_key, {})
    disease_name = info.get('name', 'Unknown')
    date = datetime.now().strftime('%Y-%m-%d')
    
    # Generate QR code
    qr_data = generate_qr_data(disease_key, confidence)
    qr_img = qrcode.QRCode(version=1, box_size=10, border=5)
    qr_img.add_data(qr_data)
    qr_img.make(fit=True)
    qr_pil = qr_img.make_image(fill_color="black", back_color="white")
    qr_buffer = io.BytesIO()
    qr_pil.save(qr_buffer, format='PNG')
    qr_buffer.seek(0)
    
    # Header with QR code
    header_data = [
        [Paragraph("<b>Skin Disease Detection Report</b>", 
                   ParagraphStyle('HeaderTitle', parent=styles['Heading1'], 
                                fontSize=22, textColor=colors.HexColor('#0066FF'), 
                                alignment=TA_LEFT)),
         RLImage(qr_buffer, width=1.2*inch, height=1.2*inch)]
    ]
    
    header_table = Table(header_data, colWidths=[5*inch, 1.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    
    story.append(header_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Date: {date}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph(f"<b>Detected Condition:</b> {disease_name}", styles['Heading2']))
    story.append(Paragraph(f"<b>Confidence Level:</b> {confidence*100:.1f}%", styles['Normal']))
    story.append(Spacer(1, 15))
    
    if info.get('description'):
        story.append(Paragraph("<b>Description</b>", styles['Heading3']))
        story.append(Paragraph(info['description'], styles['Normal']))
        story.append(Spacer(1, 15))
    
    if info.get('suggestion'):
        story.append(Paragraph("<b>Medical Advice</b>", styles['Heading3']))
        story.append(Paragraph(info['suggestion'], styles['Normal']))
        story.append(Spacer(1, 15))
    
    if info.get('cure_days'):
        story.append(Paragraph("<b>Treatment Duration</b>", styles['Heading3']))
        story.append(Paragraph(info['cure_days'], styles['Normal']))
        story.append(Spacer(1, 20))
    
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], 
                                     fontSize=10, textColor=colors.grey)
    story.append(Spacer(1, 30))
    story.append(Paragraph("Note: This is an AI-generated report. Please consult a dermatologist for professional medical advice.", 
                          disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        pred_label, confidence = predict_image(image)
        
        # Get disease info
        info = disease_info.get(pred_label, {})
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': pred_label,
            'confidence': confidence,
            'disease_name': info.get('name', 'Unknown'),
            'description': info.get('description', ''),
            'suggestion': info.get('suggestion', ''),
            'cure_days': info.get('cure_days', ''),
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    data = request.json
    disease_key = data.get('prediction')
    confidence = data.get('confidence')
    
    try:
        qr_data = generate_qr_data(disease_key, confidence)
        qr_buffer = create_qr_code(qr_data)
        
        return send_file(
            qr_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name='qr_code.png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_route():
    data = request.json
    disease_key = data.get('prediction')
    confidence = data.get('confidence')
    
    try:
        pdf_buffer = generate_pdf(disease_key, confidence)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'skin_disease_report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load model immediately (for both local and Render)
print("Loading model...")
if not load_model():
    raise RuntimeError("❌ Failed to load model. Ensure 'pure_model.tflite' exists in the project root.")

# Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"✅ Model loaded. Running on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
