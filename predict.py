from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Layer
import os

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alz_effnet_clean.keras")
model = None


class CastLayer(Layer):
    def __init__(self, **kwargs):
        super(CastLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.cast(x, tf.float32)

    def get_config(self):
        return super(CastLayer, self).get_config()


def load_alzheimer_model():
    global model
    if model is None:
        print("Loading model from:", MODEL_PATH)
        print("File exists?", os.path.exists(MODEL_PATH))
        try:
            model = load_model(
                MODEL_PATH,
                custom_objects={"Cast": CastLayer},
                compile=False,
            )
            print("✅ Model loaded successfully!")
            print("Model input shape:", model.input_shape)  # expect (None, 224, 224, 3)
        except Exception as e:
            print(f"❌ Model load error: {e}")
            model = None
    return model


def is_brain_mri_like(pil_img: Image.Image) -> bool:
    """
    Simple heuristic: reject clearly non-brain images.
    Not perfect, but filters obvious wrong uploads.
    """

    # Grayscale + downscale
    img = pil_img.convert("L").resize((64, 64))
    arr = np.array(img).astype("float32") / 255.0

    # 1) Intensity range (avoid blank images)
    intensity_range = arr.max() - arr.min()
    if intensity_range < 0.1:
        return False

    # 2) Center vs border brightness (brain in center)
    h, w = arr.shape
    center = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    border = np.concatenate([
        arr[0:h // 8, :].ravel(),
        arr[-h // 8:, :].ravel(),
        arr[:, 0:w // 8].ravel(),
        arr[:, -w // 8:].ravel(),
    ])
    center_mean = center.mean()
    border_mean = border.mean()
    if center_mean < border_mean + 0.03:
        return False

    # 3) Edge-like variation (avoid super-smooth/cartoon images)
    edge_like = np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()
    if edge_like < 0.02:
        return False

    return True


def preprocess_image(image_path):
    try:
        # Open original image and handle EXIF rotation
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        # Center-crop to square to avoid extreme distortion
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        # Brain MRI check
        if not is_brain_mri_like(img):
            print("❌ Not a brain-like MRI image")
            return None, "Uploaded image does not look like a brain MRI. Please upload a valid brain MRI scan."

        # Convert to RGB (training used RGB EfficientNet)
        img = img.convert("RGB")

        # Resize to 224x224 (model input size)
        img = img.resize((224, 224))

        # Convert to numpy and preprocess
        arr = np.array(img).astype("float32")        # (224, 224, 3)
        arr = preprocess_input(arr)                  # same as training

        # Add batch dimension
        img_array = np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)
        print("✅ Preprocessed shape:", img_array.shape)
        return img_array, None
    except Exception as e:
        print("Preprocess error:", e)
        return None, "Image preprocessing failed"


def predict_alzheimer(image_path):
    try:
        model = load_alzheimer_model()
        if model is None:
            return {"success": False, "message": "Model not loaded"}

        img_array, err = preprocess_image(image_path)
        if img_array is None:
            # err explains why (non-brain MRI or preprocess fail)
            return {"success": False, "message": err}

        print("Input shape to model.predict:", img_array.shape)
        prediction = model.predict(img_array, verbose=0)
        print("Prediction shape:", prediction.shape)
        print("Prediction vector:", prediction[0], "sum=", prediction[0].sum())

        # Order must match train_gen.class_indices:
        # {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}
        class_names = [
            "Mild Demented",       # index 0
            "Moderate Demented",   # index 1
            "Non-Demented",        # index 2
            "Very Mild Demented",  # index 3
        ]

        predicted_class_idx = int(np.argmax(prediction[0]))
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])

        class_probs = {
            class_name: float(prediction[0][i])
            for i, class_name in enumerate(class_names)
        }

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "classes": class_probs,
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"success": False, "message": str(e)}


def get_ai_suggestions(prediction_class):
    suggestions = {
        "Non-Demented": (
            "Patient shows no signs of cognitive impairment. Maintain regular "
            "cognitive health practices: healthy diet, exercise, social engagement, "
            "mental stimulation."
        ),
        "Very Mild Demented": (
            "Patient shows subtle cognitive changes. Recommend: Regular neurology "
            "check-ups, cognitive exercises, memory training, lifestyle modifications."
        ),
        "Mild Demented": (
            "Patient shows noticeable cognitive decline. Recommend: Specialist "
            "neurology consultation, cognitive behavioral therapy, family support, "
            "medication review."
        ),
        "Moderate Demented": (
            "Patient shows significant cognitive and functional impairment. Recommend: "
            "Immediate specialist care, comprehensive neuropsychological evaluation, "
            "care planning with family, potential medication adjustment."
        ),
    }
    return suggestions.get(prediction_class, "Consult with healthcare provider.")
