#!/usr/bin/env python3
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import joblib
import librosa
import numpy as np
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configure application
application = Flask(__name__)
application.secret_key = 'your-secret-key-here'
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioClassifier:
    """Encapsulates the audio classification pipeline."""
    
    def __init__(self, model_path: str = "svm_model.pkl", 
                 scaler_path: str = "scaler.pkl"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self._model = None
        self._scaler = None
    
    def _load_artifacts(self) -> Tuple[SVC, StandardScaler]:
        """Lazy load the trained model and scaler."""
        if self._model is None or self._scaler is None:
            try:
                self._model = joblib.load(str(self.model_path))
                self._scaler = joblib.load(str(self.scaler_path))
                logger.info("Model and scaler loaded successfully")
            except FileNotFoundError as e:
                logger.error(f"Model files not found: {e}")
                raise
        return self._model, self._scaler
    
    def extract_audio_features(self, filepath: Path, 
                             mfcc_count: int = 13,
                             fft_window: int = 2048,
                             step_size: int = 512) -> Optional[np.ndarray]:
        """Extract MFCC features from audio file."""
        try:
            signal, sample_rate = librosa.load(str(filepath), sr=None)
            mfcc_matrix = librosa.feature.mfcc(
                y=signal, 
                sr=sample_rate,
                n_mfcc=mfcc_count,
                n_fft=fft_window,
                hop_length=step_size
            )
            # Compute mean across time dimension
            return np.mean(mfcc_matrix.T, axis=0)
        except Exception as e:
            logger.error(f"Feature extraction failed for {filepath}: {e}")
            return None
    
    def predict_authenticity(self, audio_file_path: Path) -> str:
        """Classify audio file as genuine or deepfake."""
        if not audio_file_path.exists():
            return "Error: Audio file not found on server."
        
        if audio_file_path.suffix.lower() != '.wav':
            return "Error: Only WAV format is supported for analysis."
        
        features = self.extract_audio_features(audio_file_path)
        if features is None:
            return "Error: Could not extract features from the audio file."
        
        try:
            model, scaler = self._load_artifacts()
            normalized_features = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(normalized_features)[0]
            
            classification = "authentic" if prediction == 0 else "synthetic (deepfake)"
            return f"Audio classification result: {classification}"
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error: Classification failed due to model issues."


class FileValidator:
    """Handles file validation and security checks."""
    
    ALLOWED_EXTENSIONS = {'wav'}
    MAX_FILENAME_LENGTH = 255
    
    @classmethod
    def is_valid_audio_file(cls, filename: str) -> bool:
        """Check if filename has valid audio extension."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS and
                len(filename) <= cls.MAX_FILENAME_LENGTH)
    
    @classmethod
    def secure_file_handling(cls, uploaded_file) -> Tuple[bool, str]:
        """Validate and secure the uploaded file."""
        if not uploaded_file or not uploaded_file.filename:
            return False, "No file was selected for upload."
        
        if not cls.is_valid_audio_file(uploaded_file.filename):
            return False, "Invalid file type. Please upload a WAV audio file."
        
        return True, "File validation passed."


# Initialize classifier
classifier = AudioClassifier()
validator = FileValidator()


@application.route("/", methods=["GET", "POST"])
def home():
    """Main route for file upload and classification."""
    if request.method == "GET":
        return render_template("index.html")
    
    # Handle POST request
    uploaded_file = request.files.get("audio_file")
    is_valid, message = validator.secure_file_handling(uploaded_file)
    
    if not is_valid:
        flash(message, "error")
        return render_template("index.html", error_message=message)
    
    # Process the valid file
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            secured_name = secure_filename(uploaded_file.filename)
            temp_path = Path(temp_file.name)
            uploaded_file.save(str(temp_path))
            
            # Classify the audio
            result = classifier.predict_authenticity(temp_path)
            
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
            
            return render_template("result.html", classification_result=result)
            
    except Exception as e:
        logger.error(f"File processing error: {e}")
        error_msg = "An error occurred while processing your audio file."
        return render_template("index.html", error_message=error_msg)


@application.errorhandler(413)
def file_too_large(error):
    """Handle file size limit exceeded."""
    return render_template("index.html", 
                         error_message="File too large. Maximum size is 16MB."), 413


@application.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal error: {error}")
    return render_template("index.html", 
                         error_message="Internal server error occurred."), 500


def create_app():
    """Application factory pattern."""
    return application


if __name__ == "__main__":
    logger.info("Starting deepfake audio detection web application")
    application.run(debug=True, host="127.0.0.1", port=5000)
