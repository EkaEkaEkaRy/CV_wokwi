from ultralytics import YOLO
import cv2
import numpy as np
import requests
import io

server_url = "https://bookish-pancake-59vwq54jgv7cv74v-8000.app.github.dev"

def image_to_1bit_bytes(image):
    # Преобразуем в grayscale, если нужно
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применяем бинаризацию
    _, bw_image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    height, width = bw_image.shape
    packed_bytes = bytearray()

    # Упаковываем каждый 8 пикселей по одному байту, построчно
    for y in range(height):
        byte = 0
        bit_count = 0
        for x in range(width):
            bit = bw_image[y, x]
            byte = (byte << 1) | bit
            bit_count += 1
            if bit_count == 8:
                packed_bytes.append(byte)
                byte = 0
                bit_count = 0
        # Если остались биты, дополняем справа нулями
        if bit_count > 0:
            byte = byte << (8 - bit_count)
            packed_bytes.append(byte)
    return bytes(packed_bytes)


def send_trigger_image(image_bytes):
    files = {'image': ('frame.jpg', image_bytes, 'image/jpeg')}
    response = requests.post(f"{server_url}/trigger", files=files)
    if response.status_code == 200:
        print("Trigger sent:", response.json())
    else:
        print("Error sending trigger:", response.status_code)


model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Можно уменьшить размер перед подачей в модель для fps
    resized_frame = cv2.resize(frame, (640, 384))  # или frame.shape[1]/2, frame.shape[0]/2

    # Передаем кадр в модель
    results = model(resized_frame)[0]

    # Проверка обнаруженных объектов
    for box in results.boxes:
        cls_id = int(box.cls[0])  # класс объекта (ID)
        class_name = results.names[cls_id]  # имя класса
        if class_name == "person":
            print("Обнаружен объект: apple")
            # Координаты бокса, округляем и конвертируем в int
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Вырезаем изображение яблока из кадра
            apple_image = frame[y1:y2, x1:x2]
            resized_apple_image = cv2.resize(apple_image, (128, 64))
            resized_apple_image = resized_apple_image.astype(np.uint8)
            packed_bytes = image_to_1bit_bytes(resized_apple_image)
            send_trigger_image(packed_bytes)


    # Рисуем обнаружения поверх исходного кадра (можно использовать results.plot(),
    # но иногда лучше самому нарисовать рамки)
    img = results.plot()

    # Конвертируем из RGB (если plot возвращает RGB) в BGR для OpenCV
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('YOLOv8 Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
