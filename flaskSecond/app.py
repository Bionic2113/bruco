import imghdr
import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('first_market_model.h5')
model_bathrooms = tf.keras.models.load_model('3pk_second_model.h5')
class_names = ['Автотовары', 'Ванная', 'Для взрослых',  'Игрушки']
bathroom_names = ['Бритва', 'Ëршик', 'Зубная щётка', 'Зубная электрощетка',
                  'Мочалка', 'Сиденье для унитаза', 'Электро-бритва']
image_arr = None

# Маршрут для отображения страницы с формой
@app.route('/')
def index():
    print("Hello from index()")
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Получение файла из формы
    photo = request.files['photo']

    # Проверка, что файл был загружен
    if photo.filename == '':
        return 'Файл не выбран'

    # Создание временного файла для сохранения загруженного изображения
    temp_filepath = 'temp.jpg'
    photo.save(temp_filepath)

    # Чтение изображения с использованием OpenCV
    image = cv2.imread(temp_filepath)

    # Удаление временного файла
    os.remove(temp_filepath)

    # Проверка, что изображение было успешно прочитано
    if image is None:
        return 'Ошибка чтения изображения'

    # Изменение размера изображения с использованием OpenCV
    resized_image = cv2.resize(image, (320, 320))
    # Конвертация изображения в RGB формат
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Возвращение ответа
    predictions = predict(resized_image_rgb)
    top_predictions = get_top_predictions(predictions, class_names, 3)
    return top_predictions


@app.route(f'/process/{class_names[0]}', methods=['GET'])
def auto_goods():
    return [class_names[0], "В разработке:)"]


@app.route(f'/process/{class_names[1]}', methods=['GET'])
def bathroom():
    print("bathroom")
    return [class_names[1], bathroom_names[np.argmax(model_bathrooms.predict(image_arr).tolist()[0])]]


@app.route(f'/process/{class_names[2]}', methods=['GET'])
def for_Old():
    return [class_names[2], "В разработке:)"]


@app.route(f'/process/{class_names[3]}', methods=['GET'])
def toys():
    return [class_names[3], "В разработке:)"]


def predict(picture):
    global image_arr
    image_arr = np.array(picture) / 255.0  # Normalize the image
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension
    predictions = model.predict(image_arr).tolist()  # Convert ndarray to list
    print(predictions)
    return predictions[0]  # Return the first element of the predictions list


def get_top_predictions(predictions, classes, top_k):
    top_indices = np.argsort(predictions)[::-1][:top_k]  # Get indices of top predictions
    top_predictions = [classes[i] for i in top_indices]  # Get class names of top predictions
    return top_predictions


if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='localhost', port=8089)
