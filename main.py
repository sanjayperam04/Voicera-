#!/usr/bin/env python3
import logging
import joblib
import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class MFCCExtractor:
    """Convert a WAV file into a fixed-size MFCC feature vector."""
    def __init__(self,
                 n_mfcc: int = 13,
                 frame_length: int = 2048,
                 hop: int = 512):
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop = hop

    def extract(self, filepath: Path) -> np.ndarray:
        try:
            audio, sr = librosa.load(str(filepath), sr=None)
        except Exception as e:
            logging.warning(f"Could not load {filepath.name}: {e}")
            return None

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop
        )
        # Average over time frames → 1-D vector
        return np.mean(mfcc.T, axis=0)


def collect_features(dirs: Dict[int, Path],
                     extractor: MFCCExtractor
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan each directory in `dirs`, extract MFCCs, and return
    stacked feature matrix X and label vector y.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []

    for label, folder in dirs.items():
        wav_files = list(folder.glob("*.wav"))
        logging.info(f"Found {len(wav_files)} files in {folder}")
        for wav in wav_files:
            vec = extractor.extract(wav)
            if vec is not None:
                features.append(vec)
                labels.append(label)

    if not features:
        raise RuntimeError("No valid audio features were extracted.")
    return np.vstack(features), np.array(labels)


def train_and_save(X: np.ndarray,
                   y: np.ndarray,
                   model_path: Path,
                   scaler_path: Path,
                   test_ratio: float = 0.2
                  ) -> None:
    """
    Split X/y (if each class has ≥2 examples), scale, train linear SVM,
    evaluate on held-out data when possible, and save model + scaler.
    """
    classes, counts = np.unique(y), np.bincount(y)
    logging.info(f"Label distribution: {dict(zip(classes, counts))}")

    # Decide whether to hold out a test set
    if len(classes) == 2 and np.all(counts >= 2):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=42,
            stratify=y
        )
        logging.info(f"Train/Test split: {X_tr.shape[0]} / {X_te.shape[0]}")
    else:
        logging.warning("Not enough samples for stratified split. Using all data for training.")
        X_tr, y_tr = X, y
        X_te, y_te = None, None

    scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = scaler.transform(X_tr)

    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_tr_scaled, y_tr)
    logging.info("SVM training complete.")

    if X_te is not None:
        preds = clf.predict(scaler.transform(X_te))
        acc = accuracy_score(y_te, preds)
        cm = confusion_matrix(y_te, preds)
        logging.info(f"Test Accuracy: {acc:.2%}")
        logging.info(f"Confusion Matrix:\n{cm}")

    # Persist artifacts
    joblib.dump(clf, str(model_path))
    joblib.dump(scaler, str(scaler_path))
    logging.info(f"Saved model to {model_path} and scaler to {scaler_path}")


def classify_file(filepath: Path,
                  model_path: Path,
                  scaler_path: Path,
                  extractor: MFCCExtractor
                 ) -> None:
    """Load saved SVM and scaler, run MFCC extraction on one file, and print label."""
    if not filepath.exists() or filepath.suffix.lower() != ".wav":
        logging.error(f"Invalid file: {filepath}")
        return

    clf = joblib.load(str(model_path))
    scaler = joblib.load(str(scaler_path))
    feat = extractor.extract(filepath)

    if feat is None:
        logging.error("Feature extraction failed.")
        return

    pred = clf.predict(scaler.transform(feat.reshape(1, -1)))[0]
    label = "genuine" if pred == 0 else "deepfake"
    print(f"Result for {filepath.name}: {label}")


def main():
    # Define your data folders here
    data_dirs = {
        0: Path("path/to/real_audio/dir"),    # genuine label
        1: Path("path/to/deepfake_audio/dir") # deepfake label
    }
    extractor = MFCCExtractor()
    X, y = collect_features(data_dirs, extractor)

    model_file = Path("svm_model.pkl")
    scaler_file = Path("scaler.pkl")
    train_and_save(X, y, model_file, scaler_file)

    # Prompt for a single file classification
    test_path = Path(input("Enter path to .wav file to classify: ").strip())
    classify_file(test_path, model_file, scaler_file, extractor)


if __name__ == "__main__":
    main()
