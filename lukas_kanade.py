import csv
from statistics import mean
import sys
import cv2
import numpy as np
import glob


folder_name = sys.argv[1]
file_extension = sys.argv[2]
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points, bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        if folder_name == "Sylvestr":
            w = 54
            h = 46
        else:
            w = 30
            h = 25
        bbox = (x, y, w, h)
        old_points = np.array([[x, y]], dtype=np.float32)
        print("point", point)

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def calculate_ap(precision, recall):
    sorted_indices = sorted(range(len(recall)), key=lambda i: recall[i])
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

point_selected = False
point = ()
bbox = ()
old_points = np.array([[]])
end = False
# Lucas kanade params
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

image_files = glob.glob(f"{folder_name}/*.{file_extension}")
gt_file = f"{folder_name}/groundtruth.txt"
background = cv2.imread(image_files[0])
first_frame = cv2.imread(image_files[0])
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=5, detectShadows=False)
cv2.imshow("Frame", first_frame)
cv2.setMouseCallback("Frame", select_point)
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

gt_bboxes = []  # ground truth bbox
pred_bboxes = []  # predicted bbox
iou_scores = []  # IoU list

while True:
    if point_selected:
        with open(gt_file, 'r') as file:
            gt_data = csv.reader(file, delimiter=',')
            with open("resultsLukas.csv", 'w', newline='') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(["IoU"])

                for image_file, gt_bbox in zip(image_files, gt_data):
                    frame = cv2.imread(image_file)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mask = subtractor.apply(gray_frame)
                    mask = cv2.medianBlur(mask, 5)
                    cv2.circle(frame, point, 5, (0, 0, 255), 2)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, mask,
                                                                         **lk_params)
                    old_gray = gray_frame.copy()
                    old_points = new_points
                    x, y = new_points.ravel()
                    bbox = (int(x - bbox[2] / 2), int(y - bbox[3] / 2), bbox[2], bbox[3])
                    point2 = np.intp((x, y))
                    cv2.circle(frame, point2, 5, (0, 255, 0), -1)
                    gt_bbox = [int(float(coord)) for coord in gt_bbox]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                    iou = calculate_iou(bbox, gt_bbox)
                    iou_str = str(iou).replace('.', ',')
                    writer.writerow([iou_str])
                    iou_scores.append(iou)
                    gt_bboxes.append(gt_bbox)
                    pred_bboxes.append(bbox)
                    cv2.rectangle(frame, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, "IoU: {:.2f}".format(iou), (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Mask", mask)

                    key = cv2.waitKey(10)
                    if key == 27:
                        end = True
                        break
                break
    elif end:
        break
    key = cv2.waitKey(1)
    if key == 27:
        break
precision, recall, iou_final = calculate_precision_recall(gt_bboxes, pred_bboxes)
ap = calculate_ap(precision, recall)

print("Precision: {:.2f}". format(mean(precision)))
print("Recall: {:.2f}". format(mean(recall)))
print("AP: {:.2f}".format(ap))

with open("resultsfinalLukas.csv", 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["IoU", "Precision", "Recall", "AP"])
    writer.writerow([mean(iou_final), mean(precision), mean(recall), ap])

cv2.destroyAllWindows()