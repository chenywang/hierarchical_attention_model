# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import os

from process.components import get_sentence
from config import config


def visualize(sess, inputs, revlens, max_rev_length, keep_probs, index2word, alphas_words, alphas_sents, x_test, y_test,
              y_predict, visual_sample_index):
    visual_dir = config.visualization_path
    # visualization
    visual_file = os.path.join(visual_dir, "visualize_{}.html".format(visual_sample_index))
    x_test_sample = x_test[visual_sample_index:visual_sample_index + 1]
    y_test_sample = y_test[visual_sample_index:visual_sample_index + 1]
    test_dict = {inputs: x_test_sample, revlens: [max_rev_length], keep_probs: [1.0, 1.0]}
    alphas_words_test, alphas_sents_test = sess.run([alphas_words, alphas_sents], feed_dict=test_dict)
    y_test_predict = sess.run(y_predict, feed_dict=test_dict)
    print("test sample is {}".format(y_test_sample[0]))
    print("test sample is predicted as {}".format(y_test_predict[0]))
    print(alphas_words_test.shape)

    # visualize a review
    sents = [get_sentence(index2word, x_test_sample[0][i]) for i in range(max_rev_length)]
    index_sent = 0
    print("sents size is {}".format(len(sents)))
    with open(visual_file, "w") as html_file:
        html_file.write('actual label: %f, predicted label: %f<br>' % (y_test_sample[0], y_test_predict[0]))
        for sent, alpha in zip(sents, alphas_sents_test[0] / alphas_sents_test[0].max()):
            if len(set(sent.split(' '))) == 1:
                index_sent += 1
                continue
            visual_sent = visualize_sentence_format(sent)
            # display each sent's importance by color
            html_file.write('<font style="background: rgba(255, 0, 0, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % (alpha))
            visual_words = visual_sent.split()
            visual_words_alphas = alphas_words_test[index_sent][:len(visual_words)]
            # for each sent, display its word importance by color
            for word, alpha_w in zip(visual_words, visual_words_alphas / visual_words_alphas.max()):
                html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s </font>' % (alpha_w, word))
            html_file.write('<br>')
            index_sent += 1