from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from keras import backend as K


class FaceMatch:
    def __init__(self, model_path):
        K.clear_session()
        self.load_model()
        model.load_weights(model_path)

        self.vgg_face_descriptor = Model(inputs=model.layers[0].input
                                         , outputs=model.layers[-2].output)

    def load_model(self):
        global model
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def findCosineDistance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def comapre(self, img1, img2):
        img1_representation = self.vgg_face_descriptor.predict(self.preprocess_image(img1))[0, :]
        img2_representation = self.vgg_face_descriptor.predict(self.preprocess_image(img2))[0, :]

        cosine_similarity = self.findCosineDistance(img1_representation, img2_representation)
        # print(cosine_similarity)
        if cosine_similarity < 0.30:
            return True  # True
        else:
            return False


if __name__ == '__main__':
    api = FaceMatch('./weight/vgg_face_weights.h5')
    print(api.comapre('1.jpg', '120.jpg'))
