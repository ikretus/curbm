#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import h5py, os, sys
import numpy as np
#
########################################################################

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Converts MAT to NPY', \
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',       type=str,   default='',     help='path/to/DATA.npy')
    parser.add_argument('--h_flip',     action='store_true',        help='two-fold flip')
    parser.add_argument('--w_flip',     action='store_true',        help='double two-fold flip')
    args = parser.parse_args()

    fn = os.path.expanduser(args.data)
    fi = h5py.File(fn)

    N, S_MAX, C_MAX, H_MAX, W_MAX = fi['names'].shape[0], 8, 100, 100, 220
    names = [''.join(chr(c) for c in fi[fi['names'][i,0]][:,0]).rsplit('\\', 1)[-1] for i in xrange(N)]

    data = list()

    for i in xrange(N):
        data_i = list()
        dm = fi['volume'][i]

        hw_pos = zip(*np.where( fi['object'][i,0] >  0 ))
        hw_neg = zip(*np.where( fi['object'][i,0] == 0 ))

        pos_counter = 0
        print 'processing : pos: %d, neg: %d, %s' % (len(hw_pos), len(hw_neg), names[i])

        for h, w in hw_pos:
            h_beg, h_end = h - S_MAX, h + S_MAX + 1
            w_beg, w_end = w - S_MAX, w + S_MAX + 1
            if (h_beg >= 0) and (h_end < H_MAX) and (w_beg >= 0) and (w_end < W_MAX):
                c = dm[:, h, w].argmax()
                c_beg, c_end = c - S_MAX, c + S_MAX + 1
                if (c_beg >= 0) and (c_end < C_MAX):
                    v = dm[c_beg:c_end,h_beg:h_end,w_beg:w_end]
                    data_i.append(v.ravel())
                    if args.h_flip:
                        v1 = v[:,::-1,:]
                        data_i.append(v1.ravel())
                        if args.w_flip:
                            data_i.append( v[:,:,::-1].ravel())
                            data_i.append(v1[:,:,::-1].ravel())
                    pos_counter += 1

        np.random.shuffle(hw_neg); hw_neg = iter(hw_neg)
        while pos_counter:
            try:
                h, w = hw_neg.next()
            except StopIteration:
                break
            h_beg, h_end = h - S_MAX, h + S_MAX + 1
            w_beg, w_end = w - S_MAX, w + S_MAX + 1
            if (h_beg >= 0) and (h_end < H_MAX) and (w_beg >= 0) and (w_end < W_MAX):
                c = dm[:, h, w].argmax()
                c_beg, c_end = c - S_MAX, c + S_MAX + 1
                if (c_beg >= 0) and (c_end < C_MAX):
                    v = dm[c_beg:c_end,h_beg:h_end,w_beg:w_end]
                    data_i.append(v.ravel())
                    if args.h_flip:
                        v1 = v[:,::-1,:]
                        data_i.append(v1.ravel())
                        if args.w_flip:
                            data_i.append( v[:,:,::-1].ravel())
                            data_i.append(v1[:,:,::-1].ravel())
                    pos_counter -= 1

        print 'processed (pos+neg) : %d samples' % len(data_i)
        np.random.shuffle(data_i)
        data.extend(data_i)

    fi.close()
    print 'total vectors : %d' % len(data)

    np.save(fn[:-4] +'.npy', np.uint8(data).T)

########################################################################

