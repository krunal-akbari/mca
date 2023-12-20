from keras.layers.core import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.models import Model
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
train_dir = "Training"
test_dir = "Testing"

# preprocessing and generating training validation and testing dataset
trainDataGen = ImageDataGenerator(
    rescale=1./255,  # rescale
    validation_split=0.1,  # validation split
    zoom_range=0.2,  # zoom to create augmented sample
)
testDataGen = ImageDataGenerator(rescale=1./255)  # rescaling


trainGen = trainDataGen.flow_from_directory(train_dir,
                                            # image size
                                            target_size=(224, 224),
                                            color_mode='rgb',  # color mode of image
                                            class_mode='categorical',  # label to be categorized
                                            batch_size=128,  # specifying batch size
                                            subset='training'
                                            )
testGen = testDataGen.flow_from_directory(test_dir, target_size=(
    224, 224), color_mode='rgb', class_mode='categorical', batch_size=1)

valGen = trainDataGen.flow_from_directory(train_dir, target_size=(
    224, 224), color_mode='rgb', class_mode='categorical', batch_size=128, subset='validation')

"""##VGG16"""


base_model = VGG16(input_shape=(224, 224, 3),
                   include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.inputs, outputs=out)
model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

model.compile(loss="categorical_crossentropy",
              optimizer="Adam", metrics=["accuracy"])

steps_per_epochs = trainGen.samples//128
print(steps_per_epochs)
validation_steps_ = valGen.samples//128
print(validation_steps_)

start = time.time()
history = model.fit(trainGen, validation_data=valGen,
                    epochs=50, callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])),
         acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluating the model
model_evaluation = model.evaluate(testGen)

# Calculating accuracy
print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %")
