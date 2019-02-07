import tensorflow as tf, sys
import numpy as np
import binascii
#import cv2
# #vidcap = cv2.VideoCapture('testvideo.mkv')
# success, image = vidcap.read()
# count=0
# success = True
# dt = np.dtype(str, 10)
# while success:
#     success, image = vidcap.read()
#     cv2.imwrite("d.jpg", image)
#     image_data = tf.gfile.FastGFile("d.jpg", 'rb').read()
#     label_lines = [line.rstrip() for line in tf.gfile.GFile("temp/output_labels.txt")]

#     with tf.gfile.FastGFile("/tmp/output_graph.pb", 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         _ = tf.import_graph_def(graph_def, name='')

#     with tf.Session() as sess:
#         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#         predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

#         top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

#         for node_id in top_k:
#             human_string = label_lines[node_id]
#             score = predictions[0][node_id]
#             print('%s (score = %.5f)' %(human_string, score))
#         sess.close()

image_data = tf.gfile.FastGFile("d.jpg", 'rb').read()
label_lines = [line.rstrip() for line in tf.gfile.GFile("temp/output_labels.txt")]

with tf.gfile.FastGFile("temp/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' %(human_string, score))
        sess.close()
