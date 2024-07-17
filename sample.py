import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread('center.jpg')

# 画像をグレースケールに変換
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# エッジ検出（Canny法を使用）
edges = cv2.Canny(gray_image, 100, 200)

# 元の画像とエッジ検出後の画像を表示
plt.figure(figsize=(10, 5))

# 元の画像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# エッジ検出後の画像
plt.subplot(1, 2, 2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

# 画像を表示
plt.show()