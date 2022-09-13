from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np
angles = ['090', '000', '018', '036', '054', '072',
          '108', '126', '144', '162', '180']
pid = 75
X = []
y = []
for cond in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
    for p in range(pid, 125):
        for ang in angles:
            path = '/GaitDatasetB_gei/%03d/%s/%03d-%s-%s.png' % (
                p, cond, p, cond, ang)
            path1 = '../transformed/%03d-%s-%s.png' % (p, cond, ang)
            #print(path)
            #print(path1)
            if not os.path.exists(path):
                continue
            if ang == '090':
                img = cv2.imread(path, 0)
            else:
                img = cv2.imread(path1, 0)
            img = cv2.resize(img, (64, 64))
            img = img.flatten().astype(np.float32)
            X.append(img)
            y.append(p-pid)

nbrs = KNeighborsClassifier(n_neighbors=1, p=1, weights='distance')
X = np.asarray(X)
y = np.asarray(y).astype(np.int32)
nbrs.fit(X, y)

testX = []
testy = []
pid = 75
for cond in ['nm-05', 'nm-06']:
    for p in range(pid, 125):
        for ang in angles:
            path = '/GaitDatasetB_gei/%03d/%s/%03d-%s-%s.png' % (
                p, cond, p, cond, ang)
            path1 = '../transformed/%03d-%s-%s.png' % (p, cond, ang)
            if not os.path.exists(path):
                continue
            if ang == '090':
                img = cv2.imread(path, 0)
            else:
                img = cv2.imread(path1, 0)
            img = cv2.resize(img, (64, 64))
            img = img.flatten().astype(np.float32)
            testX.append(img)
            testy.append(p-pid)

testX = np.asarray(testX).astype(np.float32)
print(nbrs.score(testX, testy))
