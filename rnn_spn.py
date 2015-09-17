"""
File to reproduce the results for RNN-SPN

"""
from __future__ import division
import numpy as np
import theano.tensor as T
import theano
import lasagne
from repeatlayer import Repeat
from confusionmatrix import ConfusionMatrix
import os
import uuid

import logging
import argparse
np.random.seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=str, default="0.0005")
parser.add_argument("-decayinterval", type=int, default=10)
parser.add_argument("-decayfac", type=float, default=1.5)
parser.add_argument("-nodecay", type=int, default=30)
parser.add_argument("-optimizer", type=str, default='rmsprop')
parser.add_argument("-dropout", type=float, default=0.0)
parser.add_argument("-downsample", type=float, default=3.0)
args = parser.parse_args()

output_folder = "logs/RNN_SPN" + str(uuid.uuid4())[:18].replace('-', '_')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(output_folder, "results.log"), mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info('#'*80)
for name, val in sorted(vars(args).items()):
    sep = " "*(35 - len(name))
    logger.info("#{}{}{}".format(name, sep, val))
logger.info('#'*80)

np.random.seed(123)
TOL = 1e-5
num_batch = 100
dim = 100
num_rnn_units = 256
num_classes = 10
NUM_EPOCH = 300
LR = float(args.lr)
MONITOR = False
MAX_NORM = 5.0
LOOK_AHEAD = 50

org_drp = args.dropout
sh_drp = theano.shared(lasagne.utils.floatX(args.dropout))

M = T.matrix()
W_ini = lasagne.init.GlorotUniform()
W_ini_gru = lasagne.init.GlorotUniform()
W_proc_ini = lasagne.init.GlorotUniform()
W_class_init = lasagne.init.GlorotUniform()

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    mnist_sequence = "mnist_sequence3_sample_8distortions_9x9.npz"

    from lasagne.layers import dnn
    conv = dnn.Conv2DDNNLayer
    pool = lasagne.layers.MaxPool2DLayer
elif _platform == "darwin":
    mnist_sequence = "mnist_sequence3_sample_8distortions_9x9.npz"
    conv = lasagne.layers.Conv2DLayer
    pool = lasagne.layers.MaxPool2DLayer

print conv
print "Filename:", mnist_sequence

data = np.load(mnist_sequence)


x_train, y_train = data['X_train'].reshape((-1, dim, dim)), data['y_train']
x_valid, y_valid = data['X_valid'].reshape((-1, dim, dim)), data['y_valid']
x_test, y_test = data['X_test'].reshape((-1, dim, dim)), data['y_test']

Xt = x_train[:num_batch]
batches_train = x_train.shape[0] // num_batch
batches_valid = x_valid.shape[0] // num_batch

num_steps = y_train.shape[1]
sym_x = T.tensor3()
sym_y = T.imatrix()

# setup network
l_in = lasagne.layers.InputLayer((None, dim, dim))
l_dim = lasagne.layers.DimshuffleLayer(l_in, (0, 'x', 1, 2))

l_pool0_loc = pool(l_dim, pool_size=(2, 2))
l_conv0_loc = conv(l_pool0_loc, num_filters=20, filter_size=(3, 3),
                   name='l_conv0_loc', W=W_ini)
l_pool1_loc = pool(l_conv0_loc, pool_size=(2, 2))
l_conv1_loc = conv(l_pool1_loc, num_filters=20, filter_size=(3, 3),
                   name='l_conv1_loc', W=W_ini)
l_conv1_loc = lasagne.layers.DropoutLayer(l_conv1_loc, p=sh_drp)
l_pool2_loc = pool(l_conv1_loc, pool_size=(2, 2))
l_conv2_loc = conv(l_pool2_loc, num_filters=20, filter_size=(3, 3),
                   name='l_conv2_loc', W=W_ini)

l_repeat_loc = Repeat(l_conv2_loc, n=num_steps)
l_gru = lasagne.layers.GRULayer(l_repeat_loc, num_units=num_rnn_units,
                                unroll_scan=True)

l_shp = lasagne.layers.ReshapeLayer(l_gru, (-1, num_rnn_units))  # (96, 256)

b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1

# From gru hid to A
l_A_net = lasagne.layers.DenseLayer(
    l_shp,
    num_units=6,
    name='A_net',
    b=b.flatten(),
    W=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.identity)

l_conv_to_transform = lasagne.layers.ReshapeLayer(
    Repeat(l_dim, n=num_steps), [-1] + list(l_dim.output_shape[-3:]))

l_transform = lasagne.layers.TransformerLayer(
    incoming=l_conv_to_transform,
    localization_network=l_A_net,
    downsample_factor=args.downsample)

l_conv0_out = conv(l_transform, num_filters=32, filter_size=(3, 3),
                   name='l_conv0_out', W=W_ini)

l_pool1_out = pool(l_conv0_out, pool_size=(2, 2))
l_drp1_out = lasagne.layers.DropoutLayer(l_pool1_out, p=sh_drp)
l_conv1_out = conv(l_drp1_out, num_filters=32, filter_size=(3, 3),
                   name='l_conv1_out', W=W_ini)

l_pool2_out = pool(l_conv1_out, pool_size=(2, 2))
l_drp2_out = lasagne.layers.DropoutLayer(l_pool2_out, p=sh_drp)
l_conv2_out = conv(l_drp2_out, num_filters=32, filter_size=(3, 3),
                   name='l_conv2_out', W=W_ini)

#print l_pool0_out.output_shape
print l_conv0_out.output_shape
print l_conv1_out.output_shape
print l_pool1_out.output_shape
print l_pool2_out.output_shape
print l_conv2_out.output_shape


#print lasagne.layers.get_output(l_conv3_out, sym_x).eval({sym_x: Xt}).shape
#assert False

l_class1 = lasagne.layers.DenseLayer(
    l_conv2_out, num_units=400,
    W=W_class_init,
    name='class1')
l_lin_out = lasagne.layers.DenseLayer(
    l_class1, num_units=num_classes,
    W=W_class_init,
    name='class2',
    nonlinearity=lasagne.nonlinearities.softmax)
l_out = l_lin_out

output_train = lasagne.layers.get_output(
    l_out, sym_x, deterministic=False)
output_eval, l_A_eval = lasagne.layers.get_output(
    [l_out, l_A_net], sym_x, deterministic=True)



# cost
output_flat = T.reshape(output_train, (-1, num_classes))
cost = T.nnet.categorical_crossentropy(output_flat+TOL, sym_y.flatten())
cost = T.mean(cost)


all_params = lasagne.layers.get_all_params(l_out, trainable=True)
trainable_params = lasagne.layers.get_all_params(l_out, trainable=True)


for p in trainable_params:
    print p.name

all_grads = T.grad(cost, trainable_params)
all_grads = [T.clip(g, -1, 1) for g in all_grads]
sh_lr = theano.shared(lasagne.utils.floatX(LR))

# adam works with lr 0.001
updates, norm = lasagne.updates.total_norm_constraint(
    all_grads, max_norm=MAX_NORM, return_norm=True)

if args.optimizer == 'rmsprop':
    updates = lasagne.updates.rmsprop(updates, trainable_params,
                                      learning_rate=sh_lr)
elif args.optimizer == 'adam':
    updates = lasagne.updates.adam(updates, trainable_params,
                                   learning_rate=sh_lr)


if MONITOR:
    add_output = all_grads + updates.values()

    f_train = theano.function([sym_x, sym_y], [cost, output_train, norm
                                               ] + add_output,
                              updates=updates)
else:
    f_train = theano.function([sym_x, sym_y], [cost, output_train, norm],
                              updates=updates)
f_eval = theano.function([sym_x],
                         [output_eval, l_A_eval.reshape((-1, num_steps, 6))])

best_valid = 0
look_count = LOOK_AHEAD
cost_train_lst = []
last_decay = 0
for epoch in range(NUM_EPOCH):
    # eval train
    shuffle = np.random.permutation(x_train.shape[0])

    if epoch < 5:
        sh_drp.set_value(lasagne.utils.floatX((epoch)*org_drp/5.0))
    else:
        sh_drp.set_value(lasagne.utils.floatX(org_drp))

    for i in range(batches_train):
        idx = shuffle[i*num_batch:(i+1)*num_batch]
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        train_out = f_train(x_batch, y_batch)
        cost_train, _, train_norm = train_out[:3]

        if MONITOR:
            print str(i) + "-"*44 + "GRAD NORM  \t UPDATE NORM \t PARAM NORM"
            all_mon = train_out[3:]
            grd_mon = train_out[:len(all_grads)]
            upd_mon = train_out[len(all_grads):]
            for pm, gm, um in zip(trainable_params, grd_mon, upd_mon):
                if '.b' not in pm.name:
                    pad = (40-len(pm.name))*" "
                    print "%s \t %.5e \t %.5e \t %.5e" % (
                        pm.name + pad,
                        np.linalg.norm(gm),
                        np.linalg.norm(um),
                        np.linalg.norm(pm.get_value())
                    )

        cost_train_lst += [cost_train]

    conf_train = ConfusionMatrix(num_classes)
    for i in range(x_train.shape[0] // 1000):
        probs_train, _ = f_eval(x_train[i*1000:(i+1)*1000])
        preds_train_flat = probs_train.reshape((-1, num_classes)).argmax(-1)
        conf_train.batch_add(
            y_train[i*1000:(i+1)*1000].flatten(),
            preds_train_flat
        )

    if last_decay > args.decayinterval and epoch > args.nodecay:
        last_decay = 0
        old_lr = sh_lr.get_value(sh_lr)
        new_lr = old_lr / args.decayfac
        sh_lr.set_value(lasagne.utils.floatX(new_lr))
        print "Decay lr from %f to %f" % (float(old_lr), float(new_lr))
    else:
        last_decay += 1

    # valid
    conf_valid = ConfusionMatrix(num_classes)
    for i in range(batches_valid):
        x_batch = x_valid[i*num_batch:(i+1)*num_batch]
        y_batch = y_valid[i*num_batch:(i+1)*num_batch]
        probs_valid, _ = f_eval(x_batch)
        preds_valid_flat = probs_valid.reshape((-1, num_classes)).argmax(-1)
        conf_valid.batch_add(
            y_batch.flatten(),
            preds_valid_flat
        )

    # test
    conf_test = ConfusionMatrix(num_classes)
    batches_test = x_test.shape[0] // num_batch
    all_y, all_preds = [], []
    for i in range(batches_test):
        x_batch = x_test[i*num_batch:(i+1)*num_batch]
        y_batch = y_test[i*num_batch:(i+1)*num_batch]
        probs_test, A_test = f_eval(x_batch)
        preds_test_flat = probs_test.reshape((-1, num_classes)).argmax(-1)
        conf_test.batch_add(
            y_batch.flatten(),
            preds_test_flat
        )

        all_y += [y_batch]
        all_preds += [probs_test.argmax(-1)]

    logger.info(
        "Epoch {} Acc Valid {}, Acc Train = {}, Acc Test = {}".format(
            epoch,
            conf_valid.accuracy(),
            conf_train.accuracy(),
            conf_test.accuracy())
    )

    np.savez(os.path.join(output_folder, "res_test"),
             probs=probs_test, preds=probs_test.argmax(-1),
             x=x_batch, y=y_batch, A=A_test,
             all_y=np.vstack(all_y),
             all_preds=np.vstack(all_preds))

    if conf_valid.accuracy() > best_valid:
        best_valid = conf_valid.accuracy()
        look_count = LOOK_AHEAD
    else:
        look_count -= 1

    if look_count <= 0:
        break