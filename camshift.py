import cv2
import numpy as np
import csv
import glob


def calculate_iou(bbox1, bbox2):
    # Oblicz współrzędne punktów prostokąta
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Oblicz współrzędne punktów przecięcia prostokątów
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Oblicz powierzchnię przecięcia i sumę powierzchni obu prostokątów
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Oblicz współczynnik IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


image_files = glob.glob("dino/*.jpg")  # Ścieżka do katalogu z obrazami
gt_file = "dino/groundtruth.txt"  # Ścieżka do pliku gt.txt

first_frame = cv2.imread(image_files[0])

roi = cv2.selectROI(first_frame)
print('Selected bounding boxes: {}'.format(roi[0]))
#x = 193
#y = 30
#width = 89
#height = 73
x = roi[0]
y = roi[1]
width = roi[2]
height = roi[3]
expected = [x, y, width, height]
print(expected)
roi = first_frame[y: y + height, x: x + width]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0, 106, 75)), np.array((20, 208, 211)))
#kernel = np.ones((5, 5), np.uint8)
#mask = cv2.erode(mask, kernel, iterations=2)
#mask = cv2.dilate(mask, kernel, iterations=2)

#roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
#roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
iou_scores = []  # Lista do przechowywania wyników IoU

#otwierórz zdjęcie
with open(gt_file, 'r') as file:
    gt_data = csv.reader(file, delimiter=',')

    with open("results1.csv", 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["IoU"])
    #dla każdego zdjęcia i danych ground truth bboxa
        for image_file, gt_bbox in zip(image_files, gt_data):

            frame = cv2.imread(image_file)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #mask2 = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            mask2 = cv2.inRange(hsv, np.array((0, 139, 155)), np.array((24, 217, 219)))
            ret, track_window = cv2.CamShift(mask2, (x, y, width, height), term_criteria)
            pts = cv2.boxPoints(ret)
            pts = np.intp(pts)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            #x, y, w, h = track_window
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(first_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            # Oblicz IoU dla bieżącej klatki i zapisz wynik
            x = np.min(pts[:, 0])
            y = np.min(pts[:, 1])
            w = np.max(pts[:, 0]) - x
            h = np.max(pts[:, 1]) - y
            current_bbox = [x, y, w, h]
            print(current_bbox)
            gt_bbox = [int(coord) for coord in gt_bbox]
            #gt_bbox [2] = 82
            #gt_bbox [3] = 86
            cv2.rectangle(frame, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (255, 0, 0), 2)
            iou = calculate_iou(current_bbox, gt_bbox)
            iou_str = str(iou).replace('.', ',')
            # Zapisz nazwę pliku i wynik IoU do pliku CSV
            writer.writerow([iou_str])  # Zapisz nazwę pliku w pierwszej kolumnie
            iou_scores.append(iou)
            cv2.putText(frame, "IoU: {:.2f}".format(iou), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Mask", mask2)
            cv2.imshow("Frame", frame)
            cv2.imshow("First frame", first_frame)

            key = cv2.waitKey(60)
            if key == 27:
                break

cv2.destroyAllWindows()