from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import base64
from PIL import ImageOps
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
try:
    model = tf.keras.models.load_model('mrwhite.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Define your Roman numeral classes in the same order as your model's output
ROMAN_CLASSES = ['I', 'II', 'III', 'IV', 'IX', 'V', 'VI', 'VII', 'VIII', 'X']
def crop_whitespace(image):
    """
    Crop whitespace from the image to center the content
    """
    # Convert to numpy array for easier processing
    img_array = np.array(image)
    
    # Check if the image is already grayscale
    if len(img_array.shape) > 2:
        # Convert to grayscale if it's not
        gray_img = image.convert('L')
        img_array = np.array(gray_img)
    
    # Find non-white rows and columns
    # For an inverted image (white on black), we'd look for non-black
    # For a normal image (black on white), we look for non-white
    # Assuming white is close to 255 and content is darker
    rows = np.where(np.min(img_array, axis=1) < 240)[0]
    cols = np.where(np.min(img_array, axis=0) < 240)[0]
    
    # If the image is empty or completely white, return original
    if len(rows) == 0 or len(cols) == 0:
        logger.warning("Image appears to be blank - no content to crop")
        return image
    
    # Determine the boundaries for cropping
    top, bottom = rows[0], rows[-1] + 1
    left, right = cols[0], cols[-1] + 1
    
    # Add a small padding
    padding = max(3, int(min(img_array.shape) * 0.05))  # At least 3 pixels or 5% of image size
    
    top = max(0, top - padding)
    bottom = min(img_array.shape[0], bottom + padding)
    left = max(0, left - padding)
    right = min(img_array.shape[1], right + padding)
    
    # Crop the image
    cropped_img = image.crop((left, top, right, bottom))
    
    # Log the crop dimensions
    logger.debug(f"Cropped from {image.size} to {cropped_img.size}")
    logger.debug(f"Crop boundaries: top={top}, bottom={bottom}, left={left}, right={right}")
    
    return cropped_img

def preprocess_image(image_data):
    """
    Preprocess the image data for model prediction
    """
    try:
        # Convert base64 image data to PIL Image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Debug: Save the original image
        image.save('debug_original.png')
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Crop whitespace to center the content
        image = crop_whitespace(image)
        
        # Debug: Save the cropped image
        image.save('debug_cropped.png')
        
        # Debug: Save the inverted image
        image.save('debug_inverted.png')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Debug: Save the final processed input image
        image.save('debug_input.png')
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array.reshape(1, 28, 28, 1)
        image_array = image_array.astype('float32') / 255.0
        
        logger.debug(f"Processed image shape: {image_array.shape}")
        return image_array
    
    except Exception as e:
        logger.error(f"Error in preprocessing image: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        logger.debug("Received prediction request")
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction[0])
        predicted_roman = ROMAN_CLASSES[predicted_index]
        confidence = float(prediction[0][predicted_index])
        
        logger.debug(f"Prediction: {predicted_roman}, Confidence: {confidence}")
        
        return jsonify({
            'prediction': predicted_roman,
            'confidence': confidence
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)