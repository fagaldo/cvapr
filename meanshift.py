import cv2
import csv
import glob
from statistics import mean
import sys


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


def main(folder_name, file_extension):
    image_files = glob.glob(f"{folder_name}/*.{file_extension}")
    gt_file = f"{folder_name}/groundtruth.txt"
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=7, detectShadows=False)

    if folder_name == "Sylvestr":
        x = 119
        y = 59
        width = 54
        height = 46
    else:
        x = 139.12
        y = 98.164
        width = 30.65
        height = 25.347

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    gt_bboxes = []
    pred_bboxes = []
    iou_scores = []

    with open(gt_file, 'r') as file:
        gt_data = csv.reader(file, delimiter=',')

        with open("resultsMeanshift.csv", 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["IoU"])

            for image_file, gt_bbox in zip(image_files, gt_data):

                frame = cv2.imread(image_file)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = subtractor.apply(frame_gray)
                mask = cv2.medianBlur(mask, 3)
                _, track_window = cv2.meanShift(mask, (int(x), int(y), int(width), int(height)), term_criteria)
                x, y, w, h = track_window
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                current_bbox = [x, y, w, h]
                gt_bbox = [int(float(coord)) for coord in gt_bbox]
                cv2.rectangle(frame, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                              (255, 0, 0), 2)
                iou = calculate_iou(current_bbox, gt_bbox)
                iou_str = str(iou).replace('.', ',')
                writer.writerow([iou_str])
                iou_scores.append(iou)
                gt_bboxes.append(gt_bbox)
                pred_bboxes.append(current_bbox)
                cv2.putText(frame, "IoU: {:.2f}".format(iou), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2)
                cv2.imshow("Mask", mask)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

    precision, recall, iou_final = calculate_precision_recall(gt_bboxes, pred_bboxes)
    ap = calculate_ap(precision, recall)

    print("Precision: {:.2f}", mean(precision))
    print("Recall: {:.2f}", mean(recall))
    print("AP: {:.2f}".format(ap))

    with open("resultsfinalMeanshift.csv", 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["IoU", "Precision", "Recall", "AP"])
        writer.writerow([mean(iou_final), mean(precision), mean(recall), ap])

    cv2.destroyAllWindows()


folder_name = sys.argv[1]
file_extension = sys.argv[2]
main(folder_name, file_extension)
