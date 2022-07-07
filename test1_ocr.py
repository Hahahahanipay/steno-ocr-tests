from PIL import Image
# import csv
import cv2
import numpy as np
import os
from collections import Counter

TRAIN_DATA = "dataset/train/"
TRAIN_LABELS = TRAIN_DATA
TEST_DATA = "dataset/test/"
TEST_LABELS = TEST_DATA
k = 5


# def read_csv_file(path):
#     rows = []
#     if path == TRAIN_DATA:
#         path = TRAIN_LABELS
#     elif path == TEST_DATA:
#         path = TEST_LABELS
#     with open(path, 'r') as file:
#         csvreader = csv.reader(file)
#         header = next(csvreader)
#         for row in csvreader:
#             rows.append(row)
#     return rows


def load_images(path):
    images = os.listdir(path)
    return [
        resize_image(Image.open(path + img))
        for img in images
    ]
    # images = read_csv_file(path)
    # return [
    #     resize_image(Image.open(path + i[0]))
    #     for i in images
    # ]


def load_labels(path):
    labels = os.listdir(path)
    return [
        label[0: label.find('.png')]
        for label in labels
    ]
    # labels = read_csv_file(path)
    # return [
    #     i[1]
    #     for i in labels
    # ]


def resize_image(img):
    img = img.resize((28, 28))
    # img.show()
    img_array = np.array(img)
    th, img_th = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
    return img_th


def flatten(img):
    return img.flatten()


def get_features(images):
    return [
        flatten(img)
        for img in images
    ]


def distance(train_sample, test_sample):
    return sum(
        [
            (int(train_i) - int(test_i)) ** 2
            for train_i, test_i in zip(train_sample, test_sample)
        ]
    ) ** 0.5


def get_euclidian_distance(X_train, test_sample):
    return [
        distance(train_sample, test_sample)
        for train_sample in X_train
    ]


def get_nearest_neighbors(distances, k):
    return sorted(
        range(len(distances)),
        key=lambda i: distances[i]
    )[:k]


def get_k_labels(neighbors, labels):
    return [
        labels[i]
        for i in neighbors
    ]


def knn(X_train, y_train, X_test, k):
    predicted_labels = []
    distances = []
    for test_sample in X_test:
        distances.append(get_euclidian_distance(X_train, test_sample))
        nearest_neighbors = get_nearest_neighbors(distances, k)
        k_labels = get_k_labels(nearest_neighbors, y_train)
        predicted_labels.append(Counter(k_labels).most_common(1)[0][0])


def main():
    X_train = load_images(TRAIN_DATA)
    y_train = load_labels(TRAIN_LABELS)
    # y_test = load_labels(TEST_LABELS)
    X_test = load_images(TEST_DATA)

    X_train = get_features(X_train)
    X_test = get_features(X_test)

    knn(X_train, y_train, X_test, k)



if __name__ == '__main__':
    main()
