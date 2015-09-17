__author__ = 'sorensonderby'
import numpy as np
from load_mnist import load_data
from PIL import Image
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import os
ORG_SHP = [28, 28]
OUT_SHP = [100, 100]
NUM_DISTORTIONS = 8
dist_size = (9, 9)  # should be odd?
NUM_DISTORTIONS_DB = 100000

mnist_data = load_data('../MNIST/mnist.pkl.gz')
outfile = "mnist_sequence3_sample_8dsistortions9x9"

np.random.seed(1234)


### create list with distortions
all_digits = np.concatenate([mnist_data['X_train'],
                             mnist_data['X_valid'],
                             mnist_data['X_train']], axis=0)
all_digits = all_digits.reshape([-1] + ORG_SHP)
num_digits = all_digits.shape[0]


distortions = []
for i in range(NUM_DISTORTIONS_DB):
    rand_digit = np.random.randint(num_digits)
    rand_x = np.random.randint(ORG_SHP[1]-dist_size[1])
    rand_y = np.random.randint(ORG_SHP[0]-dist_size[0])

    digit = all_digits[rand_digit]
    distortion = digit[rand_y:rand_y + dist_size[0],
                       rand_x:rand_x + dist_size[1]]
    assert distortion.shape == dist_size
    #plt.imshow(distortion, cmap='gray')
    #plt.show()
    distortions += [distortion]
print "Created distortions"


def plot_n_by_n_images(images,epoch=None,folder=None, n = 10, shp=[28,28]):
    """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
    the images so that they appear reasonably close together. The
    image is post-processed to give the appearance of being continued."""
    #image = np.concatenate(images, axis=1)
    i = 0
    a,b = shp
    img_out = np.zeros((a*n, b*n))
    for x in range(n):
        for y in range(n):
            xa, xb = x*a, (x+1)*b
            ya, yb = y*a, (y+1)*b
            im = np.reshape(images[i], (a,b))
            img_out[xa:xb, ya:yb] = im
            i += 1
    # matshow(img_out*100.0, cmap = matplotlib.cm.binary)
    img_out = (255*img_out).astype(np.uint8)
    img_out = Image.fromarray(img_out)
    if folder is not None and epoch is not None:
        img_out.save(os.path.join(folder,epoch + ".png"))
    return img_out


def create_sample(x, output_shp, num_distortions=NUM_DISTORTIONS):
    a, b = x[0].shape
    x_offset = (output_shp[1]-len(x)*a)//2

    x_offset += np.random.choice(range(-x_offset, x_offset))
    y_offset = np.random.choice(range(output_shp[1]))

    angle = np.random.choice(range(int(-b*0.5), int(b*0.5)))

    output = np.zeros(output_shp)
    for i,digit in enumerate(x):
        x_start = i*b + x_offset

        x_end = x_start + b
        y_start = y_offset + np.floor(i*angle)
        y_end = y_start + a
        if y_end > (output_shp[1]-1):
            m = output_shp[1] - y_end
            y_end += m
            y_start += m
        if y_start < 0:
            m = y_start
            y_end -= m
            y_start -= m
        output[y_start:y_end, x_start:x_end] = digit

    if num_distortions > 0:
            output = add_distortions(output, num_distortions)
    return output


def sample_digits(n, x, y, out_shp=None):

    if out_shp is None:
        shp = x.shape[1]
    else:
        shp = out_shp
    n_samples = x.shape[0]
    idxs = np.random.choice(range(n_samples), replace=True, size=n)
    return [x[i].reshape(shp) for i in idxs], [y[i] for i in idxs]


def add_distortions(digits, num_distortions):
    canvas = np.zeros_like(digits)
    for i in range(num_distortions):
        rand_distortion = distortions[np.random.randint(NUM_DISTORTIONS_DB)]
        rand_x = np.random.randint(OUT_SHP[1]-dist_size[1])
        rand_y = np.random.randint(OUT_SHP[0]-dist_size[0])
        canvas[rand_y:rand_y+dist_size[0],
               rand_x:rand_x+dist_size[1]] = rand_distortion
    canvas += digits

    return np.clip(canvas, 0, 1)


def create_dataset(n, X, labels, org_shp, out_shp):
    out_X, out_lab = [], []
    out_X = np.zeros((n, np.prod(out_shp)))
    for i in range(n):
        if (i+1) % 1000 == 0:
            print i
        x_, y_ = sample_digits(3, X, labels, org_shp)
        digits = create_sample(x_, out_shp)

        digits = digits.reshape(-1)
        y_ = np.array(y_)
        out_X[i, ] = digits
        out_lab.append(y_)

    return out_X.astype('float32'), np.vstack(out_lab).astype('int32')


# generate excample picture
samples = [create_sample(sample_digits(
    3, mnist_data['X_train'],
    mnist_data['y_train'],
    ORG_SHP)[0], OUT_SHP).reshape(-1) for i in range(400)]
samples_arr = np.vstack(samples)
plot_n_by_n_images(samples_arr,epoch="0",folder="", n=20, shp=OUT_SHP)

N_TRAIN = 60000
N_VALID = 10000
N_TEST = 10000
X_train, y_train = create_dataset(N_TRAIN, mnist_data['X_train'],
                                  mnist_data['y_train'], ORG_SHP, OUT_SHP
                                  )
X_valid, y_valid = create_dataset(N_VALID, mnist_data['X_valid'],
                                  mnist_data['y_valid'], ORG_SHP, OUT_SHP
                                  )
X_test, y_test = create_dataset(N_TEST, mnist_data['X_test'],
                                mnist_data['y_test'], ORG_SHP, OUT_SHP
                                )
np.savez_compressed(
    outfile,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test)
## create train, valid, and test sets




