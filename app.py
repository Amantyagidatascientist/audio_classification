import dill
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import numpy as np
import noisereduce  as nr




with open('model/feature_extractor_dill.pkl', 'rb') as f:
    extractor = dill.load(f)

print(extractor)


model = load_model('model/my_trained_model.h5')
audio_file = "99710-9-0-12.wav"
features = extractor.transform([audio_file])
print(features)

label_mapping = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music'
    }


def predict_audio_class(file_path, extractor, model):
    features = extractor.transform([file_path])
    print(features)
    
    prediction = model.predict(features)
    print(prediction)
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    
    
    return label_mapping.get(predicted_class, "Unknown")


predicted_label = predict_audio_class(audio_file, extractor, model)
print("Predicted class:", predicted_label)

