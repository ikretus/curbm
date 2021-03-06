#!/usr/bin/env python
import argparse
import cPickle
import gzip
import os
import sys
import time
import numpy as np
import scipy.sparse as spsp
from sklearn.feature_extraction.text import TfidfTransformer
#
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
#
MAX_CPU_FLOATS = 2**30
MAX_GPU_MEMORY = 2**30
CUDAMAT_INIT = True
#
try:
    import cudamat as cm
except:
    print '\nCUDAMAT NOT LOADED'
    print 'script will be stopped\n'
    CUDAMAT_INIT = False
#
########################################################################

def lr(lr_init, n_epochs, epoch):
    return (lr_init if (epoch < n_epochs / 4) else \
            lr_init * 0.99**(700 * (float(epoch) / n_epochs - 1./4)) )

def mom(mom_init, n_epochs, epoch):
    return (mom_init if (epoch < n_epochs / 4) else 0.9)

def sigmoid(x):
    return 1./(1.+ np.exp(-x))

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=0)

def transform(W, b, v):
    return sigmoid(np.dot(W.T, v) + b)

def reconstruct(W, a, b, v):
    h = transform(W, b, v)
    return sigmoid(np.dot(W, h) + a)

def loadWeights(fn):
    fi = np.load(fn)
    return (np.float64(fi['W']), np.float64(fi['a']), np.float64(fi['b']))

def free_energy(W, a, b, v):
    return (np.dot(a.T, v) + np.log(1 + np.exp(np.dot(W.T, v) + b)).sum(0))

def plot(W, fn='plot.png'):
    isz, sz = int(np.sqrt(np.ceil(np.sqrt(W.shape[0]))**2)), int(np.sqrt(W.shape[1]))
    if W.shape[0] != isz * isz:
        v = np.vstack((W, np.zeros((isz * isz - W.shape[0], W.shape[1]))))
    else:
        v = W
    v = (v - v.min()) / (v.max() - v.min())
    fbank = np.hstack( \
                np.hstack(( \
                    np.vstack(np.vstack((v[:,i + sz*k].reshape(isz, isz), np.zeros((20, isz)))) for i in xrange(sz)), \
                    np.zeros((sz*(isz + 20), 20)) )) for k in xrange(sz) )
    pl.imsave(fn, fbank, cmap=pl.cm.Greys_r, dpi=300)

########################################################################

if __name__=='__main__':
    p = argparse.ArgumentParser(description='RBM training via cudamat', \
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch',    type=int,   default=64,     help='batch size')
    p.add_argument('--epochs',   type=int,   default=1000,   help='number of epochs')
    p.add_argument('--data',     type=str,   default=None,   help='path/to/data.NPY')
    p.add_argument('--dropout',  action='store_true',        help='dropout half of hiddens')
    p.add_argument('--gpu',      type=int,   default=0,      help='gpu device id')
    p.add_argument('--vis',      type=str,   default='sigm', help='sigm|smax|relu|spls|linr')
    p.add_argument('--hid',      type=str,   default='sigm', help='sigm|smax|relu|spls|linr')
    p.add_argument('--lr',       type=float, default=0.01,   help='initial lr')
    p.add_argument('--l1',       type=float, default=1e-4,   help='L1 weight decay')
    p.add_argument('--l2',       type=float, default=1e-4,   help='L2 weight decay')
    p.add_argument('--mom',      type=float, default=0.9,    help='initial momentum')
    p.add_argument('--norm',     type=str,   default=None,   help='sparse vectors norm: l1, l2 or None')
    p.add_argument('--nhid',     type=int,   default=256,    help='number of hiddens')
    p.add_argument('--scale',    type=float, default=255,    help='scale of dense data')
    p.add_argument('--snapshot', type=int,   default=25,     help='snapshot iterstep')
    p.add_argument('--shuffle',  action='store_true',        help='shuffle after each epoch')
    p.add_argument('--sparse',   action='store_true',        help='data is sparse.NPZ')
    p.add_argument('--test',     action='store_true',        help='validate on 5%%-subset')
    p.add_argument('--trans',    type=str,   default=None,   help='path/to/trans.NPY')
    p.add_argument('--weights',  type=str,   default=None,   help='path/to/weights.NPZ')
    args = p.parse_args()

####CUDAMAT initialization

    if not CUDAMAT_INIT:
        sys.exit(1)
    else:
        start_time = time.time()
        cm.cuda_set_device(args.gpu)
        cm.cublas_init()
        cm.CUDAMatrix.init_random( int(time.time()) )

####Load data

    if not args.sparse:
        data = np.load(os.path.expanduser(args.data))
    else:
        d_transformer = TfidfTransformer(norm=args.norm, sublinear_tf=True)
        t_transformer = TfidfTransformer(norm=args.norm, sublinear_tf=True)

        fi = np.load(os.path.expanduser(args.data))
        dc = spsp.coo_matrix((fi['dc_data'], (fi['dc_row'], fi['dc_col']))).tocsc()
        tc = spsp.coo_matrix((fi['tc_data'], (fi['tc_row'], fi['tc_col']))).tocsc()

        data = spsp.csc_matrix( \
                    spsp.hstack((spsp.csc_matrix(fi['age']).T, \
                                 spsp.csc_matrix(fi['sex']).T, \
                                 d_transformer.fit_transform(dc), \
                                 t_transformer.fit_transform(tc))).T, dtype=np.float32)
        del dc, tc
        cPickle.dump((d_transformer, t_transformer), gzip.open('transformer_%d.gz' % start_time, 'w'))
        print 'loaded data shape = (%d, %d)' % data.shape

####Train/test split

    if args.test:
        shuffle_idx = np.arange(data.shape[1])
        np.random.shuffle(shuffle_idx)
        data = data[:,shuffle_idx]

        dlen = int(0.95 * data.shape[1])
        if not args.sparse:
            test = data[:,dlen:].astype('float32') / args.scale
        else:
            test = data[:,dlen:].toarray()
        data = data[:,:dlen]

####Load weights and/or transform data

    if args.weights:
        W, a, b = loadWeights(os.path.expanduser(args.weights))
        if args.trans:
            if not args.sparse:
                trans_data = transform(W, b, data.astype('float32') / args.scale)
                np.save(os.path.expanduser(args.trans), trans_data)
            else:
                batches = list()
                mem_batches = 1 + int(np.floor(float(np.prod(data.shape)) / MAX_CPU_FLOATS))
                vec_step = float(data.shape[1]) / mem_batches
                if vec_step == np.floor(vec_step):
                    vec_step = int(vec_step)
                else:
                    vec_step = int(np.floor(vec_step))
                    mem_batches += 1
                for mem_batch in xrange(mem_batches):
                    beg = vec_step * mem_batch
                    end = vec_step * (mem_batch + 1)
                    if beg >= data.shape[1]: break
                    if end >= data.shape[1]: end = data.shape[1]
                    batch = data[:,beg:end].toarray()
                    print 'batch %d from %d, shape = (%d, %d)\n' % ((1 + mem_batch, mem_batches) + batch.shape)
                    batches.append( transform(W, b, batch) )
                trans_data = np.hstack(batches)
                np.save(os.path.expanduser(args.trans), trans_data)
            print 'data transformation done: new shape = (%d, %d)' % trans_data.shape
            sys.exit(0)

####Initialization

    print '\n==========< %d >==========\n' % start_time

    n_vis = data.shape[0]
    n_hid = args.nhid if not args.weights else W.shape[1]
    print 'number of samples = %d' % data.shape[1]
    print 'vis(%s) = %d -> hid(%s) = %d' % (args.vis, n_vis, args.hid, n_hid)
    print 'dtype = %s, minval = %f, maxval = %f' % (data.dtype, data.min(), data.max())

    total_batches = data.shape[1] / args.batch
    print 'total integer number of batches = %d' % total_batches

    mem_batches = 4.0 * np.prod(data.shape) / MAX_GPU_MEMORY
    if mem_batches == np.floor(mem_batches):
        mem_batches = int(mem_batches)
    else:
        mem_batches = 1 + int(np.floor(mem_batches))
    print 'number of mem-batches = %d, each size = %d' % (mem_batches, MAX_GPU_MEMORY)

    n_batches = total_batches / mem_batches
    print 'batches per mem-batch = %d' % n_batches
    print 'vectors per mem-batch = %d' % (n_batches * args.batch)

    shuffle_idx = np.arange(data.shape[1])

    if not args.weights:
        W = cm.CUDAMatrix( 0.01 * np.random.randn(n_vis, n_hid) )
        a = cm.CUDAMatrix( np.zeros((n_vis, 1)) )
        b = cm.CUDAMatrix( -4.0 * np.ones((n_hid, 1)) )
    else:
        W = cm.CUDAMatrix(W)
        a = cm.CUDAMatrix(a)
        b = cm.CUDAMatrix(b)

    dW = cm.CUDAMatrix( np.zeros((n_vis, n_hid)) )
    da = cm.CUDAMatrix( np.zeros((n_vis,     1)) )
    db = cm.CUDAMatrix( np.zeros((n_hid,     1)) )

    v = cm.empty((n_vis, args.batch))
    h = cm.empty((n_hid, args.batch))
    r = cm.empty((n_hid, args.batch))

#####Main train cycle

    for epoch in xrange(1, 1 + args.epochs):
        print 'Epoch %d' % epoch
        train_err = list()
        for mem_batch in xrange(mem_batches):
            beg = n_batches * args.batch * mem_batch
            end = n_batches * args.batch * (mem_batch + 1)
            if beg >= data.shape[1]: break
            if end >  data.shape[1]: end = data.shape[1]
            if not args.sparse:
                dev_data = cm.CUDAMatrix(data[:,beg:end].astype('float32') / args.scale)
            else:
                dev_data = cm.CUDAMatrix(data[:,beg:end].toarray())
            for batch in xrange(n_batches):
                curr_lr  = args.lr    #lr(args.lr, args.epochs, epoch)
                curr_mom = args.mom   #mom(args.mom, args.epochs, epoch)

                v_true = dev_data.slice(batch * args.batch, (batch + 1) * args.batch)
                v.assign(v_true)

                dW.mult(curr_mom)
                da.mult(curr_mom)
                db.mult(curr_mom)

                # h <- act(b + W.T * v)
                cm.dot(W.T, v, target=h)
                h.add_col_vec(b)
                if args.hid == 'sigm':
                    h.apply_sigmoid()
                elif args.hid == 'smax':
                    h.apply_softmax()
                elif args.hid == 'relu':
                    h.apply_relu()
                elif args.hid == 'spls':
                    h.apply_softplus()

                if args.dropout:
                    h.dropout(0.5)

                dW.add_dot(v, h.T)
                da.add_sums(v, axis=1)
                db.add_sums(h, axis=1)

                # h <- sample(r < h)
                r.fill_with_rand()
                r.less_than(h, target=h)

                # v <- act(a + W * h)
                cm.dot(W, h, target=v)
                v.add_col_vec(a)
                if args.vis == 'sigm':
                    v.apply_sigmoid()
                elif args.vis == 'smax':
                    v.apply_softmax()
                elif args.vis == 'relu':
                    v.apply_relu()
                elif args.vis == 'spls':
                    v.apply_softplus()

                # h <- act(b + W.T * v)
                cm.dot(W.T, v, target=h)
                h.add_col_vec(b)
                if args.hid == 'sigm':
                    h.apply_sigmoid()
                elif args.hid == 'smax':
                    h.apply_softmax()
                elif args.hid == 'relu':
                    h.apply_relu()
                elif args.hid == 'spls':
                    h.apply_softplus()

                dW.subtract_dot(v, h.T)
                if args.l1 > 0:
                    dW.add_mult_sign(W, mult=args.l1)
                if args.l2 > 0:
                    dW.add_mult(W, mult=args.l2)
                da.add_sums(v, axis=1, mult=-1.)
                db.add_sums(h, axis=1, mult=-1.)

                W.add_mult(dW, curr_lr / args.batch)
                a.add_mult(da, curr_lr / args.batch)
                b.add_mult(db, curr_lr / args.batch)

                v.subtract(v_true)
                train_err.append(float(v.euclid_norm()**2 / args.batch))

            print '\t%d from %d\t%f\t%f' % (1 + mem_batch, mem_batches, np.mean(train_err), np.std(train_err))
            dev_data.free_device_memory()

####Validation

        if args.test:
            test_recon = reconstruct(W.asarray(), a.asarray(), b.asarray(), test)
            test_err = np.sum((test - test_recon)**2) / test.shape[1]

        print 'time: %d\tmean(err): %f\tstd(err): %f\tlr: %f\tmom: %f\ttest_error: %f' % \
            (time.time() - start_time, np.mean(train_err), np.std(train_err), curr_lr, curr_mom, test_err)

####Shuffling

        if args.shuffle:
            np.random.shuffle(shuffle_idx)
            data = data[:,shuffle_idx]

####Snapshotting

        if epoch % args.snapshot == 0:
            print '\nsnapshotting at epoch %d\n' % epoch
            signature = (args.vis, args.hid, n_vis, n_hid, start_time, epoch)
            np.savez('Wab_%s_%s_%d-%d-%d_%d.npz' % signature, W=W.asarray(), a=a.asarray(), b=b.asarray())
            plot(W.asarray(), 'Wab_%s_%s_%d-%d-%d_%d.png' % signature)

    cm.cublas_shutdown()

########################################################################


