import tensorflow as tf
import os
# step 1:
# Create the list of images from the relative dir: store string of path to list [filenames] and label(dir)[label]

directory = './images'

images = [] # [x[0] for x in os.walk(directory)]
labels = []
classifications = []
i = 0
for path, subdirs, files in os.walk(directory):
    query = ''
    for name in files:
        images.append(name)
        labels.append(i)
        query = name.split('_')[0]
    i+=1
    if query:
        classifications.append(query) # need to make this shit work properly


print(images)
print(labels)
print(classifications)

filenames = tf.constant(images)
labels = tf.constant(labels)

# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# step 3: parse every image in the dataset using `map`
# can we force resizing?
def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

# REPLACE WITH FULL ALADDIN PERRSON (modify folder naming to be variable, data size, batchsize, train/total, validation/total, etc)
