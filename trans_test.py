import numpy as np
from PIL import Image
from util.trans_function import RGB2HSV, HSV2RGB
import matplotlib.pyplot as plt

img_path = r'E:\hhj\SYSU-MM01\cam1\0001\0008.jpg'
img = Image.open(img_path)
img = img.resize((144, 288), Image.Resampling.LANCZOS)
img_np = np.array(img)
img_gray = img.convert('L')
print(img_np)

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title('init_RGB')

img_hsv = RGB2HSV(img_np)

img_rgb = HSV2RGB(img_hsv)
print(img_rgb * 255.0)

plt.subplot(1, 3, 2)
plt.imshow(img_rgb)
plt.title('trans_RGB')

plt.subplot(1, 3, 3)
plt.imshow(img_gray, cmap='gray')
plt.title('gray')
plt.show()
# Compare this snippet from data_test.py:
