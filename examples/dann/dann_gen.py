'''
This is the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin

This allows domain adaptation (when you want to train on a dataset
with different statistics than a target dataset) in an unsupervised manner
by using the adversarial paradigm to punish features that help discriminate
between the datasets during backpropagation.

This is achieved by usage of the 'gradient reversal' layer to form
a domain invariant embedding for classification by an MLP.

The example here uses the 'MNIST-M' dataset as described in the paper.

Credits:
- Clayton Mellina (https://github.com/pumpikano/tf-dann) for providing
  a sketch of implementation (in TF) and utility functions.
- Yusuke Iwasawa (https://github.com/fchollet/keras/issues/3119#issuecomment-230289301)
  for Theano implementation (op) for gradient reversal.

Author: Vanush Vaswani (vanush@gmail.com)
'''

from __future__ import print_function

import argparse
import os
import pickle
import sys

import mpl_toolkits.axisartist as AA
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.manifold import TSNE

import keras.backend as K
from keras.datasets import mnist
from keras.datasets import mnist_m
from keras.engine.training import make_batches
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GradientReversal
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.utils.visualize_util import plot

if K.backend() == "tensorflow":
    def set_tf_session(gpu_fraction=0.3):
        '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            print("THREADS====================")
            session = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads,
                allow_soft_placement=True, log_device_placement=False))
        else:
            print("NO THREADAS ===============")
            session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                       allow_soft_placement=True, log_device_placement=False))

        K.set_session(session)

    import tensorflow as tf

parser = argparse.ArgumentParser(description="args parser")
parser.add_argument("run_name", help="")
parser.add_argument("gpu_fraction", type=float, help="")

args = parser.parse_args()
run_name = args.run_name
if not os.path.exists(run_name):
    os.makedirs(run_name)
gpu_fraction = args.gpu_fraction

if K.backend() == "tensorflow":
    print("gpu fraction ", args.gpu_fraction)
    set_tf_session(gpu_fraction=gpu_fraction)

if sys.argv[1] :
    run_name = sys.argv[1] 
else :
    print("no run_name arugment")
save_plt_figure = True


# Helper functions

def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        # The AxesGrid object work as a list of axes.
        grid[i].imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def batch_gen(batches, id_array, data, labels):
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = id_array[batch_start:batch_end]
        if labels is not None:
            yield data[batch_ids], labels[batch_ids]
        else:
            yield data[batch_ids]
        np.random.shuffle(id_array)


def evaluate_dann(num_batches, size):
    acc = 0
    for i in range(0, num_batches):
        _, prob = dann_src_model.predict_on_batch(XT_test[i * size:i * size + size])
        predictions = np.argmax(prob, axis=1)
        actual = np.argmax(y_test[i * size:i * size + size], axis=1)
        acc += float(np.sum((predictions == actual))) / size
    return acc / num_batches


# Model parameters

batch_size = 128
nb_epoch = 200
nb_classes = 10
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 5

_TRAIN = K.variable(1, dtype='uint8')

# Prep source data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Prep target data
mnistm = mnist_m.load_data()
XT_test = np.swapaxes(np.swapaxes(mnistm[b'test'], 1, 3), 2, 3)
XT_train = np.swapaxes(np.swapaxes(mnistm[b'train'], 1, 3), 2, 3)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = np.concatenate([X_train, X_train, X_train], axis=1)
X_test = np.concatenate([X_test, X_test, X_test], axis=1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

XT_train = XT_train.astype('float32')
XT_test = XT_test.astype('float32')
XT_train /= 255
XT_test /= 255

domain_labels = np.vstack([np.tile([0, 1], [batch_size / 2, 1]),
                           np.tile([1., 0.], [batch_size / 2, 1])])

# Created mixed dataset for TSNE visualization
num_test = 500
combined_test_imgs = np.vstack([X_test[:num_test], XT_test[:num_test]])
combined_test_labels = np.vstack([y_test[:num_test], y_test[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                 np.tile([0., 1.], [num_test, 1])])


class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD()
        #self.opt = Adam()

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            activation='relu')(model_input)
        net = Convolution2D(nb_filters, nb_conv, nb_conv,
                            activation='relu')(net)
        net = MaxPooling2D(pool_size=(nb_pool, nb_pool))(net)
        net = Dropout(0.5)(net)
        net = Flatten(name="feature_1")(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu', name="cls_dense_1")(model_input)
        net = Dropout(0.5)(net)
        net = Dense(nb_classes, activation='softmax',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=net)
        if plot_model:
            plot(model, show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        self.grl = GradientReversal(1.0)
        branch = self.grl(net)
        branch = Dense(128, activation='relu', name="source_dense_1")(branch)
        branch = Dropout(0.1)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch)

        net = self._build_classifier(net)
        #model for source data
        model_src = Model(input=main_input, output=[branch, net])
        #model for target data
        model_tgt = Model(input=main_input, output=[branch])

        model_src.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])

        model_tgt.compile(loss={'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'] )

        if plot_model:
            plot(model_src, show_shapes=True)
            plot(model_tgt, show_shapes=True)
        return (model_src, model_tgt)

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(input=main_input,
                           output=self.domain_invariant_features)
        return tsne_model


main_input = Input(shape=(3, img_rows, img_cols), name='main_input')

builder = DANNBuilder()
src_model = builder.build_source_model(main_input)
src_vis = builder.build_tsne_model(main_input)

dann_src_model, dann_tgt_model = builder.build_dann_model(main_input)
dann_vis = builder.build_tsne_model(main_input)
final_gr = K.gradients(dann_src_model.output, main_input)


print('Training source only model')
# src_model.fit(X_train, y_train, batch_size=64, nb_epoch=10, verbose=2,
#               validation_data=(X_test, y_test))
print('Evaluating target samples on source-only model')
print('Accuracy: ', src_model.evaluate(XT_test, y_test,verbose=2)[1])

# Broken out training loop for a DANN model.
src_index_arr = np.arange(X_train.shape[0])
target_index_arr = np.arange(XT_train.shape[0])

batches_per_epoch = len(X_train) / batch_size
num_steps = nb_epoch * batches_per_epoch
j = 0

print('Training DANN model')

print("src model metric names ", dann_src_model.metrics_names)
print("tgt model metric names ", dann_tgt_model.metrics_names)

metric_src_epoch_list = []
metric_tgt_epoch_list = []

gf_names = ["cls_dense_1", "classifier_output", "source_dense_1", "domain_output"]
gf_list = [K.function([main_input, K.learning_phase()],
                      K.gradients(dann_src_model.get_layer(layer_name).get_output_at(0), [dann_src_model.get_layer("feature_1").get_output_at(0)]))
           for layer_name in gf_names]

tr_grad_log = []           
tr_mtr_log = []

for i in range(nb_epoch):
    gf_epoch_results = [[] for _ in range(len(gf_list))]

    batches = make_batches(X_train.shape[0], batch_size // 2)
    target_batches = make_batches(XT_train.shape[0], batch_size // 2)

    src_gen = batch_gen(batches, src_index_arr, X_train, y_train)
    target_gen = batch_gen(target_batches, target_index_arr, XT_train, None)

    losses = list()
    acc = list()

    print('## Epoch ', i)

    metric_src_batch_list = []
    metric_tgt_batch_list = []
    for idx, (xb, yb) in enumerate(src_gen):

        # Update learning rate and gradient multiplier as described in
        # the paper.
        p = float(j) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        builder.grl.l = l
        builder.opt.lr = lr

        if xb.shape[0] != batch_size / 2:
            continue

        try:
            xt = next(target_gen)
        except:
            # Regeneration
            target_gen = target_gen(target_batches, target_index_arr, XT_train,
                                    None)


        metric_src = dann_src_model.train_on_batch({'main_input': xb},
                                            {'classifier_output': yb,
                                            'domain_output': domain_labels[:batch_size//2]},
                                            )

        #np.ones(batch_size)
        #np.concatenate(np.ones(batch_size//2), np.zeros(batch_size//2))

        metric_src_batch_list.append(metric_src)

        metric_tgt = dann_tgt_model.train_on_batch({'main_input': xt},
                                            {'domain_output': domain_labels[batch_size//2:]},
                                            )
        metric_tgt_batch_list.append(metric_tgt)
        j += 1

        for gf_idx, f in enumerate(gf_list):
            con_x = np.concatenate((xb,xt), axis=0)
            grad = f([con_x,1])[0] # 1 for training mode
            gf_epoch_results[gf_idx].append(grad)

    
    gf_ep_raw_results = []
    gf_ep_pr_results = []
    for gfr_idx, gf_result in enumerate(gf_epoch_results):
        gf_ep = np.mean(gf_result, axis=0)
        gf_ep_raw_results.append(gf_ep)
        gf_sum = np.sum(gf_ep)
        gf_ss = np.sum(np.square(gf_ep))
        gf_ep_pr_results.append([gf_sum, gf_ss])
        
    
    gf_metric_result = list(zip(gf_names, gf_ep_pr_results))
    print(gf_metric_result)
    tr_grad_log.append(list(zip(gf_names, gf_ep_raw_results)))
    tr_mtr_log.append(gf_metric_result)

    metric_src_mean = np.mean(np.asarray(metric_src_batch_list), axis=0)
    metric_tgt_mean = np.mean(np.asarray(metric_tgt_batch_list), axis=0)

    metric_src_epoch_list.append(metric_src_mean)
    metric_tgt_epoch_list.append(metric_tgt_mean)
    print("tgt_metric", metric_tgt_mean,"\nsrc_metric", metric_src_mean)


# auxilary funcs
result_path = "D:/data/result/dann_gen"
gf_metric_path = os.path.join(result_path, "gf_metric_results.pkl")
tr_grad_path = os.path.join(result_path, "tr_grad_log.pkl")

def saveTrResult():
    pickle.dump(gf_metric_result, open(gf_metric_path, "wb"))
    pickle.dump(tr_grad_log, open(tr_grad_path, "wb"))

    
def loadTrResult():
    pickle.load(open(gf_metric_path, "rb"))    
    pickle.load(open(tr_grad_path, "rb"))    

def pltOut(path):
    if save_plt_figure :
        print("save plt fig :", path)
        plt.savefig(path)
    else :
        plt.show()

# if needs more metric
# tr_grad_log
#%%

d_acc_hist = [m[3] for m in metric_src_epoch_list]
d_loss_hist = [m[1] for m in metric_src_epoch_list]
l_acc_hist = [m[4] for m in metric_src_epoch_list]
l_loss_hist = [m[2] for m in metric_src_epoch_list]
d_out_grad_hist = [m[3][1][1] for m in tr_mtr_log]
l_out_grad_hist = [m[1][1][1] for m in tr_mtr_log]
    
#%%
# plot graph loss, acc
epr = list(range(len(metric_src_epoch_list)))  # epoch range
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(epr, d_acc_hist, 'r-', label="d-acc") # d acc
ax2.plot(epr, d_loss_hist, 'r--', label="d-loss") # d loss
ax.plot(epr, l_acc_hist, 'b-', label="l-acc") # l acc
ax2.plot(epr, l_loss_hist, 'b--', label="l-loss") # l loss

ax.set_ylim([0.0, 1.0])

plt.title("acc, loss")
legend = ax.legend(loc='upper left', shadow=True, bbox_to_anchor=(1.1,1))
legend = ax2.legend(loc='upper left', shadow=True, bbox_to_anchor=(1.1,0.7))

frame = legend.get_frame()
frame.set_facecolor('0.99')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
pltOut(os.path.join(run_name,"acc_loss.png"))
#plt.show()


#%%
# plot graph loss, acc
epr = list(range(len(metric_src_epoch_list)))  # epoch range
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(epr, d_acc_hist, 'r-', label="d-acc") # d acc
ax2.plot(epr, d_out_grad_hist, 'r--', label="d-grad") # domain_out_grad
ax.plot(epr, l_acc_hist, 'b-', label="l-acc") # l acc
ax2.plot(epr, l_out_grad_hist, 'b--', label="l-grad") # cls_out_grad

ax.set_ylim([0.0, 1.0])

plt.title("acc, grad_sqr")
legend = ax.legend(loc='upper left', shadow=True, bbox_to_anchor=(1.1,1))
legend = ax2.legend(loc='upper left', shadow=True, bbox_to_anchor=(1.1,0.7))

frame = legend.get_frame()
frame.set_facecolor('0.99')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
#plt.show()
pltOut(os.path.join(run_name,"acc_grad.png"))



#%%

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))

par2.axis["right"].toggle(all=True)

host.set_xlim(0,len(epr))

host.set_xlabel("epoch")
host.set_ylabel("acc")
par1.set_ylabel("out_grad")
par2.set_ylabel("l/d grad")

host.plot(epr, d_acc_hist, 'r-', label="d-acc") # d acc
host.plot(epr, l_acc_hist, 'b-', label="l-acc") # l acc

par1.set_yscale("log")
par1.plot(epr, d_out_grad_hist, 'r--', label="d-grad") # domain_out_grad
par1.plot(epr, l_out_grad_hist, 'b--', label="l-grad") # cls_out_grad

par2.set_yscale("log")
par2.plot(epr, np.divide(l_out_grad_hist, d_out_grad_hist), 'g--', label="l/d grad")


fig = plt.figure() 
default_size = fig.get_size_inches() 
plt.rcParams["figure.figsize"] = (7,3)

#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)

host.legend(loc='upper left', shadow=True, bbox_to_anchor=(0.0,-0.1))

plt.draw()
#plt.show()
pltOut(os.path.join(run_name,"acc_grad_gradr.png"))




#plt.plot(list(range(len(domain_loss_history))), domain_loss_history,"r-", list(range(len(cls_loss_history))), cls_loss_history,"b-")
#%%
domain_loss_history =[m[1] for m in metric_src_epoch_list]
cls_loss_history =[m[2] for m in metric_src_epoch_list]

plt.plot(list(range(len(domain_loss_history))), domain_loss_history,"r-", list(range(len(cls_loss_history))), cls_loss_history,"b-")
#plt.savefig(name)
plt.close()

#pickle.dump(metric_src_epoch_list, os.path.join(run_name, "src_metric.pkl"))


#%%
print('Evaluating target samples on DANN model')
size = batch_size // 2
nb_testbatches = XT_test.shape[0] // size
acc = evaluate_dann(nb_testbatches, size)
print('Accuracy:', acc)
print('Visualizing output of domain invariant features')

# Plot both MNIST and MNIST-M
imshow_grid(X_train)
imshow_grid(XT_train)

src_embedding = src_vis.predict([combined_test_imgs])
src_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = src_tsne.fit_transform(src_embedding)

plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'Source only')

dann_embedding = dann_vis.predict([combined_test_imgs])
dann_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = dann_tsne.fit_transform(dann_embedding)

plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'DANN')

#plt.show()
pltOut(os.path.join(run_name,"tsne.png"))
