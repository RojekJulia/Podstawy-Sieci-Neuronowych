import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def test_model(model_path='model_0.87.keras', test_dir='./photos/test', samples_per_class=10):
    model = tf.keras.models.load_model(model_path)
    
    for emotion in ['happy', 'sad']:
        path = os.path.join(test_dir, emotion)
        images = os.listdir(path)[:samples_per_class]
        
        plt.figure(figsize=(20, 4))
        correct_predictions = 0
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(path, img_name)
            
            img = tf.keras.preprocessing.image.load_img(
                img_path,
                color_mode='grayscale',
                target_size=(48, 48)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            predicted_emotion = 'happy' if predicted_class == 0 else 'sad'
            
            if predicted_emotion == emotion:
                correct_predictions += 1
            
            plt.subplot(2, samples_per_class, idx + 1)
            display_img = cv2.imread(img_path)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            plt.imshow(display_img)
            color = 'green' if predicted_emotion == emotion else 'red'
            plt.title(f'Real: {emotion}\nPred: {predicted_emotion}\nConf: {confidence:.2f}', 
                     color=color)
            plt.axis('off')
        
        accuracy = correct_predictions / samples_per_class
        print(f"\nAccuracy for {emotion}: {accuracy*100:.2f}%")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_model()