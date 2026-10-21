import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# Model configurations
MODEL_CONFIGS = {
    "ResNet50": {
        "filename": "resnet50_best.keras",
        "target_size": (224, 224),
        "preprocess_func": resnet_preprocess
    },
    "MobileNetV2": {
        "filename": "mobilenetv2_best.keras",
        "target_size": (224, 224),
        "preprocess_func": mobilenet_preprocess
    },
    "InceptionV3": {
        "filename": "inceptionv3_best.keras",
        "target_size": (299, 299),
        "preprocess_func": inception_preprocess
    },
    "DenseNet121": {
        "filename": "densenet121_best.keras",
        "target_size": (224, 224),
        "preprocess_func": densenet_preprocess
    }
}

CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

class ModelService:
    def __init__(self):
        self.models = {}
        self.is_loaded = False
        self.model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    def load_models(self):
        if self.is_loaded:
            return
            
        print("Loading models...")
        for model_name, config in MODEL_CONFIGS.items():
            model_path = os.path.join(self.model_dir, config["filename"])
            try:
                if os.path.exists(model_path):
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                    print(f"Loaded {model_name} successfully.")
                else:
                    print(f"Model file not found: {model_path}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                
        if len(self.models) == 4:
            self.is_loaded = True

    def process_image(self, image: Image.Image, target_size, preprocess_func):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Apply model-specific preprocessing
        processed_img = preprocess_func(img_array)
        return processed_img

    def predict_all(self, image: Image.Image):
        if not self.is_loaded:
            raise RuntimeError("Models are not loaded yet.")
            
        results = {}
        for model_name, config in MODEL_CONFIGS.items():
            model = self.models.get(model_name)
            if model is None:
                continue
                
            processed_input = self.process_image(image, config["target_size"], config["preprocess_func"])
            preds = model.predict(processed_input)
            
            # Assuming output is shape (1, 4)
            probabilities = preds[0].tolist()
            predicted_class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class_idx])
            
            prob_dict = {
                CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
            }
            
            results[model_name] = {
                "prediction": CLASS_NAMES[predicted_class_idx],
                "confidence": confidence,
                "probabilities": prob_dict
            }
            
        return results

model_service = ModelService()
