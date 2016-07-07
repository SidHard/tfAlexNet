import numpy as np
import os
import cv2

class Dataset:

    def __init__(self, imagePath, extensions):
        self.data = createImageList(imagePath, extensions)
        np.random.shuffle(self.data)
        self.num_records = len(self.data)
        self.next_record = 0

        self.labels, self.inputs = zip(*self.data)

        category = np.unique(self.labels)
        self.num_labels = len(category)
        self.category2label = dict(zip(category, range(len(category))))
        self.label2category = {l: k for k, l in self.category2label.items()}

        # Convert the labels to numbers
        self.labels = [self.category2label[l] for l in self.labels]

    def __len__(self):
        return self.num_records

    def onehot(self, label):
        v = np.zeros(self.num_labels)
        v[label] = 1
        return v

    def recordsRemaining(self):
        return len(self) - self.next_record

    def hasNextRecord(self):
        return self.next_record < self.num_records

    def preprocess(self, img):
        pp = cv2.resize(img, (227, 227))
        pp = np.asarray(pp, dtype=np.float32)
        pp /= 255
        pp = pp.reshape((pp.shape[0], pp.shape[1], 3))
        return pp

    def nextRecord(self):
        if not self.hasNextRecord():
            np.random.shuffle(self.data)
            self.next_record = 0
            self.labels, self.inputs = zip(*self.data)

            category = np.unique(self.labels)
            self.num_labels = len(category)
            self.category2label = dict(zip(category, range(len(category))))
            self.label2category = {l: k for k, l in self.category2label.items()}
    
            # Convert the labels to numbers
            self.labels = [self.category2label[l] for l in self.labels]
#            return None
        label = self.onehot(self.labels[self.next_record])
        input = self.preprocess(cv2.imread(self.inputs[self.next_record]))
        self.next_record += 1
        return label, input

    def nextBatch(self, batch_size):
        records = []
        for i in range(batch_size):
            record = self.nextRecord()
            if record is None:
                break
            records.append(record)
        labels, input = zip(*records)
        return labels, input

def createImageList(imagePath, extensions):
    imageFilenames = []
    labels = []
    categoryList = [None]
    categoryList = [c for c in sorted(os.listdir(imagePath))
                    if c[0] != '.' and
                    os.path.isdir(os.path.join(imagePath, c))]
    for category in categoryList:
        if category:
            walkPath = os.path.join(imagePath, category)
        else:
            walkPath = imagePath
            category = os.path.split(imagePath)[1]

        w = _walk(walkPath)
        while True:
            try:
                dirpath, dirnames, filenames = w.next()
            except StopIteration:
                break
            # Don't enter directories that begin with '.'
            for d in dirnames[:]:
                if d.startswith('.'):
                    dirnames.remove(d)
            dirnames.sort()
            # Ignore files that begin with '.'
            filenames = [f for f in filenames if not f.startswith('.')]
            # Only load images with the right extension
            filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in extensions]
            filenames.sort()
            # imageFilenames = [os.path.join(dirpath, f) for f in filenames]

            for f in filenames:
                imageFilenames.append([category, os.path.join(dirpath, f)])

    return imageFilenames

def _walk(top):
    """
    Directory tree generator lifted from python 2.6 and then
    stripped down.  It improves on the 2.5 os.walk() by adding
    the 'followlinks' capability.
    GLU: copied from image sensor
    """
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    yield top, dirs, nondirs
    for name in dirs:
        path = os.path.join(top, name)
        for x in _walk(path):
            yield x
