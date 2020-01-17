from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D
import matplotlib.pyplot as plt
def createModel():
    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    return model

def create_training_substance(train,valid,test):
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip = True
    )
    valid_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(train,
                                                     target_size=(100,100),
                                                     batch_size=32,
                                                     class_mode="categorical")
    valid_set = valid_datagen.flow_from_directory(valid,
                                                     target_size=(100, 100),
                                                     batch_size=32,
                                                     class_mode="categorical")
    test_set = test_datagen.flow_from_directory(test,
                                                     target_size=(100, 100),
                                                     batch_size=32,
                                                     class_mode="categorical")
    return training_set,valid_set,test_set

def training(train,valid,model):
    history = model.fit_generator(
        train,steps_per_epoch=1000,
        epoch=10,
        validation_data=valid,
        validation_steps=800
    )
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

model = createModel()
train,valid,test = create_training_substance(r'../ImageDetectionProject/Training',r'../ImageDetectionProject/Validation',r'../ImageDetectionProject/Validation')
training(train,valid,model)
# def test(model,test):
#     import numpy
#     from keras.preprocessing import image
#     model.predict()