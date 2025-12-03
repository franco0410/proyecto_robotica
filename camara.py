from ultralytics import YOLO
import cv2
import serial
import time

ser = serial.Serial('/dev/ttyACM0',9600,timeout=1)
time.sleep(1)
MODEL_PATH = "/home/pi/berry_detector/best.pt"

# Cargar modelo
model = YOLO(MODEL_PATH)

# Inicializar cámara
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ No puedo abrir la cámara")
    exit()

# Crear ventana UNA sola vez
cv2.namedWindow("Arandanos", cv2.WINDOW_NORMAL)

print("🎥 Cámara iniciada... presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al leer frame")
        break

    # Inferencia
    results = model(frame, verbose=False)
   
    # Dibujar resultados en la misma imagen (no crea ventanas nuevas)
    annotated = results[0].plot()

    # Mostrar SIEMPRE en la misma ventana
    cv2.imshow("Arandanos", annotated)
   
    if len(results[0].boxes) > 0:
        class_id = int(results[0].boxes[0].cls[0])
        class_name = model.names[class_id]

        if class_name == "RipeBlueBerry":
            ser.write(b'T')
        else:
            ser.write(b'F')
           
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
