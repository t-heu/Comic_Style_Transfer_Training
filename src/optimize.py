from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

tf.compat.v1.disable_eager_execution()

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):

    if slow:
        batch_size = 1

    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1,) + style_target.shape
    print("Shape do estilo:", style_shape)

    # Precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.compat.v1.Session() as sess:
        style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        X_content = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        content_net = vgg.net(vgg_path, X_pre)
        content_features = {CONTENT_LAYER: content_net[CONTENT_LAYER]}

        if slow:
            preds = tf.Variable(tf.random.normal(X_content.get_shape()) * 0.256)
            preds_pre = preds
        else:
            preds = transform.net(X_content / 255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER]) * batch_size
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(int, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :-1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :-1, :])
        tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        loss = content_loss + style_loss + tv_loss

        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.compat.v1.global_variables_initializer())

        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)

        total_batches = len(content_targets) // batch_size
        total_steps = total_batches * epochs

        for epoch in range(epochs):
            iterations = 0
            while iterations * batch_size < len(content_targets):
                start_time = time.time()

                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)

                for j, img_p in enumerate(content_targets[curr:step]):
                    X_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {X_content: X_batch}
                sess.run(train_step, feed_dict=feed_dict)

                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print(f"UID: {uid}, batch time: {delta_time:.2f}s")

                current_step = epoch * total_batches + iterations
                progress = (current_step / total_steps) * 100
                print(f"\rProgresso: [{current_step}/{total_steps}] {progress:.2f}%", end="", flush=True)

                is_print_iter = iterations % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= len(content_targets)
                should_print = is_print_iter or is_last

                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    _style_loss, _content_loss, _tv_loss, _loss, _preds = sess.run(to_get, feed_dict=feed_dict)

                    losses = (_style_loss, _content_loss, _tv_loss, _loss)

                    if slow:
                        _preds = vgg.unprocess(_preds)
                    else:
                        print("Tentando salvar checkpoint em:", save_path)
                        saver = tf.compat.v1.train.Saver()
                        res = saver.save(sess, save_path)
                        print("Checkpoint salvo em:", res)
                        print("Arquivos na pasta checkpoint:")
                        print(os.listdir(os.path.dirname(save_path)))

                    print(f"Losses -> style: {_style_loss:.4f}, content: {_content_loss:.4f}, tv: {_tv_loss:.4f}, total: {_loss:.4f}")
                    yield (_preds, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
