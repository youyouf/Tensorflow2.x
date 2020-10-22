import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("./snapshots/model_cpm.h5")


img_path = "./datasets/validation/images/p0.jpg"
img = cv2.resize(cv2.imread(img_path,1), (368,368))/255.0
img = np.expand_dims(img,0)
pred = model.predict(img)
print("pred:",len(pred))

points = []

for i in range(7):
    plt.imshow(pred[0][:,:,i])
    plt.show()
    max_index = np.where( pred[0][:,:,i] == np.max(pred[0][:,:,i]) )

    points.append(
        [max_index[1][0] * (400/46),
         max_index[0][0] * (600/46)]
    )
img = cv2.imread(img_path,1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

for item in points:
    cv2.circle(img,tuple(np.int32(np.array(item))), radius=5,color=(0,255,0), thickness=-1)

plt.imshow(img)
plt.show()



