import cv2
import os

# 保存するフォルダのパス
on_folder = "./on"
off_folder = "./off"

# フォルダが存在しない場合は作成する
if not os.path.exists(on_folder):
    os.makedirs(on_folder)
if not os.path.exists(off_folder):
    os.makedirs(off_folder)

# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(1)

# 画像のカウントを初期化
on_count = 0
off_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できません")
        break

    cv2.imshow('Webcam', frame)
    
    key = cv2.waitKey(1) & 0xFF

    # 'o'キーを押すとON画像を保存
    if key == ord('o'):
        on_count += 1
        filename = os.path.join(on_folder, f'on_{on_count}.jpg')
        cv2.imwrite(filename, frame)
        print(f"ON画像 {filename} を保存しました")
    
    # 'f'キーを押すとOFF画像を保存
    if key == ord('f'):
        off_count += 1
        filename = os.path.join(off_folder, f'off_{off_count}.jpg')
        cv2.imwrite(filename, frame)
        print(f"OFF画像 {filename} を保存しました")

    # 'q'キーを押すとプログラムを終了
    if key == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()