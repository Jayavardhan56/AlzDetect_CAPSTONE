# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.layers import Layer
# import os

# # alz_final_best.h5 is in the same folder as predict.py
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "alz_effnet_clean.keras")
# model = None


# class CastLayer(Layer):
#     def __init__(self, **kwargs):
#         super(CastLayer, self).__init__(**kwargs)

#     def call(self, x):
#         return tf.cast(x, tf.float32)

#     def get_config(self):
#         return super(CastLayer, self).get_config()


# def load_alzheimer_model():
#     global model
#     if model is None:
#         print("Loading model from:", MODEL_PATH)
#         print("File exists?", os.path.exists(MODEL_PATH))
#         try:
#             model = load_model(
#                 MODEL_PATH,
#                 custom_objects={"Cast": CastLayer},
#                 compile=False
#             )
#             print("✅ Model loaded successfully!")
#             print("Model input shape:", model.input_shape)  # expect (None, 224, 224, 3)
#         except Exception as e:
#             print(f"❌ Model load error: {e}")
#             model = None
#     return model


# def preprocess_image(image_path):
#     try:
#         # Open as grayscale
#         img = Image.open(image_path).convert("L")  # "L" = 1 channel

#         # Resize to 224x224
#         img = img.resize((224, 224))

#         # Convert to numpy and repeat to 3 channels
#         img = np.array(img)  # (224, 224)
#         img = np.repeat(img[..., np.newaxis], 3, axis=-1)  # (224, 224, 3)

#         img = img.astype("float32")
#         img = preprocess_input(img)

#         img_array = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
#         print("✅ Preprocessed shape:", img_array.shape)
#         return img_array
#     except Exception as e:
#         print("Preprocess error:", e)
#         return None



# def predict_alzheimer(image_path):
#     try:
#         model = load_alzheimer_model()
#         if model is None:
#             return {"success": False, "message": "Model not loaded"}

#         img_array = preprocess_image(image_path)
#         if img_array is None:
#             return {"success": False, "message": "Preprocessing failed"}

#         print("Input shape to model.predict:", img_array.shape)
#         prediction = model.predict(img_array, verbose=0)
#         print("Prediction shape:", prediction.shape)

#         class_names = [
#             "Non-Demented",
#             "Very Mild Demented",
#             "Mild Demented",
#             "Moderate Demented",
#         ]

#         predicted_class_idx = int(np.argmax(prediction[0]))
#         predicted_class = class_names[predicted_class_idx]
#         confidence = float(prediction[0][predicted_class_idx])

#         class_probs = {
#             class_name: float(prediction[0][i])
#             for i, class_name in enumerate(class_names)
#         }

#         return {
#             "success": True,
#             "prediction": predicted_class,
#             "confidence": confidence,
#             "classes": class_probs,
#         }

#     except Exception as e:
#         print("❌ Prediction error:", e)
#         return {"success": False, "message": str(e)}


# def get_ai_suggestions(prediction_class):
#     suggestions = {
#         "Non-Demented": (
#             "Patient shows no signs of cognitive impairment. Maintain regular "
#             "cognitive health practices: healthy diet, exercise, social engagement, "
#             "mental stimulation."
#         ),
#         "Very Mild Demented": (
#             "Patient shows subtle cognitive changes. Recommend: Regular neurology "
#             "check-ups, cognitive exercises, memory training, lifestyle modifications."
#         ),
#         "Mild Demented": (
#             "Patient shows noticeable cognitive decline. Recommend: Specialist "
#             "neurology consultation, cognitive behavioral therapy, family support, "
#             "medication review."
#         ),
#         "Moderate Demented": (
#             "Patient shows significant cognitive and functional impairment. Recommend: "
#             "Immediate specialist care, comprehensive neuropsychological evaluation, "
#             "care planning with family, potential medication adjustment."
#         ),
#     }
#     return suggestions.get(prediction_class, "Consult with healthcare provider.")

from PIL import Image
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


def preprocess_image(image_path):
    try:
        # Load as RGB (training used RGB with EfficientNetB3)
        img = Image.open(image_path).convert("RGB")  # 3 channels

        # Resize to 224x224 (training size)
        img = img.resize((224, 224))

        # Convert to numpy
        img = np.array(img).astype("float32")        # (224, 224, 3)

        # Same preprocessing as training
        img = preprocess_input(img)

        # Add batch dimension
        img_array = np.expand_dims(img, axis=0)      # (1, 224, 224, 3)
        print("✅ Preprocessed shape:", img_array.shape)
        return img_array
    except Exception as e:
        print("Preprocess error:", e)
        return None


def predict_alzheimer(image_path):
    try:
        model = load_alzheimer_model()
        if model is None:
            return {"success": False, "message": "Model not loaded"}

        img_array = preprocess_image(image_path)
        if img_array is None:
            return {"success": False, "message": "Preprocessing failed"}

        print("Input shape to model.predict:", img_array.shape)
        prediction = model.predict(img_array, verbose=0)
        print("Prediction shape:", prediction.shape)
        print("Prediction vector:", prediction[0], "sum=", prediction[0].sum())

        # Order must match train_gen.class_indices:
        # {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}
        class_names = [
            "Very Mild Demented",       # index 0
            "Moderate Demented",   # index 1
            "Non-Demented",        # index 2
            "Mild Demented",  # index 3
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
