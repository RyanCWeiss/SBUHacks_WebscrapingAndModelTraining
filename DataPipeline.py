
import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow import keras
import tensorflow.keras.layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

directory = "./images/"
img_edge_length= 100
img_height = img_edge_length
img_width = img_edge_length

# class_names = ['cat', 'dog']
class_names = ['circle', 'square']
filename = '_'.join(class_names)

#===================================================#
#                  Create Datasets                  #
#===================================================#

def create_datasets(directory, img_height,img_width):
    batch_size = 2

    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",  # categorical, int, binary
        class_names= class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,
        seed=123, # consistency between identical training inputs
        validation_split=0.1,
        subset="training",
    )
    # size = (img_height, img_width)
    # ds_train = ds_train.map(lambda img: tf.cast(img,tf.float32)/255)

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",  # catagorical, int, binary
        class_names= class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,
        seed=123, # consistency between identical training inputs
        validation_split=0.1,
        subset="validation",
    )
    # ds_validation = ds_validation.map(lambda img: tf.keras.preprocessing.image.smart_resize(img, size))

    ds_train = ds_train.map(augment)
    ds_validation = ds_validation.map(augment)

    return ds_train, ds_validation

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


ds_train, ds_validation = create_datasets(directory,img_height, img_width)

#===================================================#
#                Create/load Model                  #
#===================================================#


# do_fine_tuning = True
#
# module_selection = ("mobilenet_v2_100_224", 224) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
# handle_base, pixels = module_selection
# MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
# IMAGE_SIZE = (pixels, pixels)
# print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
#
#
# # BATCH_SIZE = 32 #@param {type:"integer"}
# BATCH_SIZE = 2
#
# model = tf.keras.Sequential([
#     hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
#     tf.keras.layers.Dropout(rate=0.2),
#     tf.keras.layers.Dense(ds_train.cardinality().numpy(), activation='softmax',
#                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))
# ])
# model.build((None,)+IMAGE_SIZE+(3,))
# model.summary()
#
# model.compile(
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
#     metrics=['accuracy'])
#
# steps_per_epoch = BATCH_SIZE # BATCH_SIZE
# validation_steps = BATCH_SIZE # BATCH_SIZE
# hist = model.fit(
#     ds_train,
#     epochs=5, steps_per_epoch=steps_per_epoch,
#     validation_data=ds_validation,
#     validation_steps=validation_steps).history

num_classes = len(class_names)

def load_model(class_names):
    sorted(class_names)
    filename = '_'.join(class_names)
    path = f"trainingmodels/{filename}.hdf5"

    try:
        model = keras.models.load_model(path)
    except:
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(
                scale=1./255, offset=0.0,
            ),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            tf.keras.layers.Conv2D(100, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(1, 1), padding='same'),
            tf.keras.layers.Conv2D(80, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(20, 3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
    return model






#===================================================#
#                    Train Model                    #
#===================================================#

def trainModel(model, ds_train, ds_validation):
    epochs=10
    history = model.fit(
        ds_train,
        validation_data=ds_train,
        epochs=epochs
    )
    return




#===================================================#
#                    Save Model                     #
#===================================================#
def save_model(class_names):
    sorted(class_names)
    filename = '_'.join(class_names)
    path = f"trainingmodels/{filename}.hdf5"
    model.save(path)
    return


#===================================================#
#                      Run                          #
#===================================================#

model = load_model(class_names)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.build()

trainModel(model, ds_train, ds_validation)

model.summary()

save_model(class_names)