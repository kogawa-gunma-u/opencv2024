import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time
from datetime import datetime
import requests

# LINE通知用の関数
def LINE_message(msg):
    url = "https://notify-api.line.me/api/notify"
    token = "XXXXXXXXXXXXXXXXXXXXX" # アクセストークンに置き換えてください
    headers = {"Authorization": "Bearer " + token}
    payload = {"message": msg}
    requests.post(url, headers=headers, params=payload)

# 画像データの読み込みと前処理
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))  # 画像を100x100にリサイズ
            images.append(resized.flatten())  # 1次元配列に変換
            labels.append(label)
    return images, labels

# 蛍光灯の画像フォルダ
on_folder = "./on"
off_folder = "./off"

# 画像とラベルのリストを作成
on_images, on_labels = load_images_from_folder(on_folder, 1)
off_images, off_labels = load_images_from_folder(off_folder, 0)

# データセットを結合
X = np.array(on_images + off_images)
y = np.array(on_labels + off_labels)

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVMの訓練
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# モデルの評価
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(1)

last_check_time = time.time()  # 最後にチェックした時間を記録
last_prediction = None  # 前回の予測を記録

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できません")
        break

    current_time = time.time()
    if current_time - last_check_time >= 60:  # 1分ごとにチェック
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))  # 画像を100x100にリサイズ
        flattened = resized.flatten().reshape(1, -1)  # 1次元配列に変換して形状を変更

        prediction = clf.predict(flattened)
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if prediction == 1 and last_prediction != 1:
            message = f"照明がついたようです　日時: {current_datetime}"
            print(message)
            LINE_message(message)
        elif prediction == 0 and last_prediction != 0:
            message = f"照明が消えたようです　日時: {current_datetime}"
            print(message)
            LINE_message(message)

        last_prediction = prediction
        last_check_time = current_time  # 最後にチェックした時間を更新

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()