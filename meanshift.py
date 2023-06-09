import cv2
import csv
import glob
from statistics import mean

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

def calculate_ap(precision, recall):
    sorted_indices = sorted(range(len(recall)), key=lambda i: recall[i])  # Indeksy posortowane rosnąco według odzysku
    sorted_recall = [recall[i] for i in sorted_indices]
    sorted_precision = [precision[i] for i in sorted_indices]

    ap = 0.0
    previous_recall = 0.0
    for i in range(len(sorted_recall)):
        if sorted_recall[i] > previous_recall:
            ap += sorted_precision[i] * (sorted_recall[i] - previous_recall)
            previous_recall = sorted_recall[i]

    return ap


def calculate_precision_recall(gt_bboxes, pred_bboxes):
    tp = 0
    fp = 0
    fn = 0

    iou_scores = []
    precision_list = []
    recall_list = []

    for gt_bbox, pred_bbox in zip(gt_bboxes, pred_bboxes):
        iou = calculate_iou(gt_bbox, pred_bbox)
        iou_scores.append(iou)

        if iou >= 0.5:
            tp += 1
        else:
            fp += 1

        fn = len(gt_bboxes) - tp

        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)

        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list, iou_scores



image_files = glob.glob("Sylvestr/*.png")  # Ścieżka do katalogu z obrazami
gt_file = "Sylvestr/groundtruth.txt"  # Ścieżka do pliku gt.txt

background = cv2.imread(image_files[0])
first_frame = cv2.imread(image_files[0])
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=7, detectShadows=False)

#roi = cv2.selectROI(first_frame)
#print('Selected bounding boxes: {}'.format(roi[0]))
x = 119
y = 59
width = 54
height = 46
#x = roi[0]
#y = roi[1]
#width = roi[2]
#height = roi[3]

roi = first_frame[y: y + height, x: x + width]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#mask = cv2.inRange(hsv_roi, np.array((0, 106, 75)), np.array((20, 208, 211)))
#kernel = np.ones((5, 5), np.uint8)
#mask = cv2.erode(mask, kernel, iterations=2)
#mask = cv2.dilate(mask, kernel, iterations=2)
#roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
#roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
gt_bboxes = []  # Lista przechowująca ground truth bbox
pred_bboxes = []  # Lista przechowująca przewidywane bbox
iou_scores = []  # Lista do przechowywania wyników IoU

with open(gt_file, 'r') as file:
    gt_data = csv.reader(file, delimiter=',')

    #dla każdego zdjęcia i danych ground truth bboxa
    with open("results.csv", 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["IoU"])

        for image_file, gt_bbox in zip(image_files, gt_data):

            frame = cv2.imread(image_file)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Odjęcie tła
            mask = subtractor.apply(frame_gray)
            # Wygładzenie maski
            mask = cv2.medianBlur(mask, 5)
            #mask2 = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            #mask2 = cv2.inRange(hsv, np.array((0, 139, 155)), np.array((24, 217, 219)))

            _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(first_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            # Oblicz IoU dla bieżącej klatki i zapisz wynik
            current_bbox = [x, y, w, h]
            gt_bbox = [int(coord) for coord in gt_bbox]
            cv2.rectangle(frame, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (255, 0, 0), 2)
            iou = calculate_iou(current_bbox, gt_bbox)
            iou_str = str(iou).replace('.', ',')
            writer.writerow([iou_str])

            iou_scores.append(iou)

            gt_bboxes.append(gt_bbox)
            pred_bboxes.append(current_bbox)

            cv2.putText(frame, "IoU: {:.2f}".format(iou), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Mask", mask)
            cv2.imshow("Frame", frame)
            cv2.imshow("First frame", first_frame)
            key = cv2.waitKey(10)
            if key == 27:
                break

precision, recall, iou_final = calculate_precision_recall(gt_bboxes, pred_bboxes)
ap = calculate_ap(precision, recall)


print("Precision: {:.2f}", mean(precision))
print("Recall: {:.2f}", mean(recall))
print("AP: {:.2f}".format(ap))

with open("resultsfinal.csv", 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["IoU", "Precision", "Recall", "AP"])
    writer.writerow([mean(iou_final), mean(precision), mean(recall), ap])

cv2.destroyAllWindows()