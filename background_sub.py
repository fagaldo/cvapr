import cv2
import numpy as np
import glob
import os

# Ścieżka do folderu z obrazami
image_folder = "Sylvestr"

# Ścieżka do folderu, w którym będą zapisywane wynikowe obrazy
output_folder = "bgsubtracted"

# Wczytywanie ścieżek do obrazów
image_files = sorted(glob.glob(image_folder + "/*.png"))

# Wczytywanie pierwszego obrazu jako tło referencyjne
background = cv2.imread(image_files[0])
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Tworzenie modelu subtractora tła
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=5, detectShadows=False)#screateBackgroundSubtractorKNN(100, 400, False)

# Tworzenie folderu wynikowego, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# Przetwarzanie kolejnych obrazów
for i, image_file in enumerate(image_files[1:], start=1):
    frame = cv2.imread(image_file)

    # Konwersja do skali szarości
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Odjęcie tła
    mask = subtractor.apply(frame_gray)

    # Wygładzenie maski
    mask = cv2.medianBlur(mask, 5)

    # Zapisywanie wynikowego obrazu
    output_file = os.path.join(output_folder, f"background{i}.jpg")
    cv2.imwrite(output_file, mask)

    # Wyświetlanie obrazu i maski
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Oczekiwanie na klawisz 'q' do zakończenia przetwarzania
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Zwalnianie zasobów i zamknięcie okien
cv2.destroyAllWindows()