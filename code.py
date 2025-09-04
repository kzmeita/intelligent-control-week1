import cv2
import numpy as np

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =========================
    # Definisi rentang warna
    # =========================
    # Merah
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # Hijau
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Biru
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # =========================
    # Fungsi bantu deteksi warna
    # =========================
    def detect_color(mask, frame, color_name, box_color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Deteksi masing-masing warna
    detect_color(mask_red, frame, "Merah", (0, 0, 255))   # BGR: merah
    detect_color(mask_green, frame, "Hijau", (0, 255, 0)) # BGR: hijau
    detect_color(mask_blue, frame, "Biru", (255, 0, 0))   # BGR: biru

    # Tampilkan hasil
    cv2.imshow("Deteksi Warna", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
