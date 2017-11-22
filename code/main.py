"""
Recognizer.

Usage:
  main.py train [--image_dir=<id> --learning_rate=<lr>]
  main.py recognize <image_path>

Options:
  -h --help             Show this screen.
  --version             Show version.
  --image_dir=<id>      Training image dir [default: images].
  --learning_rate=<lr>  Learning rate [default: 0.01].
"""
import os
import sys
import tensorflow as tf
from docopt import docopt
import retrain


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FakeNamespace(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def train(image_dir='images', learning_rate=0.01):
    settings = {
        'image_dir': image_dir,
        'learning_rate': learning_rate,
        'bottleneck_dir': 'bottlenecks',
        'eval_step_interval': 10,
        'final_tensor_name': 'final_result',
        'flip_left_right': False,
        'how_many_training_steps': 4000,
        'model_dir': 'inception',
        'output_graph': 'retrained_graph.pb',
        'output_labels': 'retrained_labels.txt',
        'print_misclassified_test_images': False,
        'random_brightness': 0,
        'random_crop': 0,
        'random_scale': 0,
        'summaries_dir': 'training_summaries/long',
        'test_batch_size': -1,
        'testing_percentage': 10,
        'train_batch_size': 100,
        'validation_batch_size': 100,
        'validation_percentage': 10}
    flags = FakeNamespace(settings)
    retrain.FLAGS = flags
    tf.app.run(main=retrain.main, argv=[])


def recognize(photo_path):

    # change this as you see fit
    image_path = photo_path

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))


if __name__ == '__main__':
    arguments = docopt(__doc__, version='1337')
    print(arguments)
    if arguments['recognize']:
        recognize(arguments['<image_path>'])
    elif arguments['train']:
        image_dir = arguments['--image_dir']
        learning_rate = float(arguments['--learning_rate'])
        train(image_dir=image_dir, learning_rate=learning_rate)
