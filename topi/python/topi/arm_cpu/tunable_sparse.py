"""Tunable sparse schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import logging

import tvm
from tvm import autotvm

from ..util import traverse_inline, get_const_tuple
from ..nn.util import get_const_int, get_pad_tuple
from ..nn import pad

from ..generic import schedule_sparse_conv2d_nchw
from ..nn import sparse_conv2d

logger = logging.getLogger('topi')

@autotvm.register_topi_compute(sparse_conv2d, 'arm_cpu', ['direct'])
def sparse_conv2d_arm_cpu(cfg, data, kernel_data, kernel_indices, kernel_indptr, kernel_size, strides, padding, dilation, layout, out_dtype):
    if layout == 'NCHW':
        return sparse_conv2d_nchw_autotvm(cfg, data, kernel_data, kernel_indices, kernel_indptr,
                                  kernel_size, strides, padding, dilation, out_dtype)
    else:
        raise ValueError("Unsupported layout {}.".format(layout))


@autotvm.register_topi_schedule(schedule_sparse_conv2d_nchw, 'arm_cpu', ['direct'])
def schedule_sparse_conv2d_nchw_arm_cpu(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])
    conv = None
    data_vec = None
    data_pad = None
    kernel_data_vec = None

    def _callback(op):
        if op.tag == 'sparse_conv2d_output' in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[2]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_data_vec = conv.op.input_tensors[3]
            if kernel_data_vec.op.name == 'kernel_data_vec':
                kernel = kernel_data_vec.op.input_tensors[0]
            else:
                kernel = kernel_data_vec

            schedule_sparse_conv2d_nchw_autotvm(cfg, s, data_vec, kernel_data_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

def sparse_conv2d_nchw_autotvm(cfg, data, kernel_data, kernel_indices, kernel_indptr,
                               kernel_size, strides, padding, dilation, out_dtype):
    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size

    if isinstance(strides, int):
        HSTR = WSTR = strides
    else:
        HSTR, WSTR = strides

    if len(kernel_data.shape) == 1:
        CO = get_const_int(kernel_indptr.shape[0] - 1)
    elif len(kernel_data.shape) == 3:
        CO = get_const_int((kernel_indptr.shape[0] - 1) * kernel_data.shape[1])
    else:
        raise ValueError("Only CSR or BSR supported")

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    OH = (IH + pad_top + pad_bottom - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])

    NNZ, BS_R, BS_C = get_const_tuple(kernel_data.shape)
    BI = BS_C // (KH * KW)
    NUM_BLKS_PLUS_1, = get_const_tuple(kernel_indptr.shape)
    NUM_BLKS = NUM_BLKS_PLUS_1 - 1    # NUM_BLKS = CO // BS_R

    # ================== define configuration space =========================
    cfg.add_flop(2*N*CO*CI*OH*OW*KH*KW)
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    bi, kh, kw = cfg.reduce_axis(BI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    nb = cfg.axis(NUM_BLKS)
    bs_r = cfg.axis(BS_R)

    nb, vc = cfg.define_split('tile_co', nb, num_outputs=2)
    oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
    ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)

    cfg.define_reorder("reorder_0",
                       [n, nb, bs_r, oh, ow, bi, kh, kw, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, nb, bs_r, oh, ow, bi, kh, kw, vh, vw, vc],
                           [n, nb, bs_r, oh, ow, bi, kh, kw, vc, vh, vw]])

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')

    VC = cfg["tile_co"].size[1]
    VH = cfg["tile_oh"].size[1]
    VW = cfg["tile_ow"].size[1]

    dvshape = (N, OH // VH, OW // VW, CI, VH*HSTR + KH-1, VW*WSTR + KW-1)
    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
                           data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw],
                           name='data_vec')

    kdvshape = (NNZ, BS_R, BI, KH, KW)
    kernel_data_vec = tvm.compute(kdvshape, lambda nnz, bs_r, bi, kh, kw:
                                  kernel_data[nnz][bs_r][bi*KH*KW + kh*KW + kw],
                                  name='kernel_data_vec')

    bi = tvm.reduce_axis((0, BI), name="bi")
    kh = tvm.reduce_axis((0, KH), name="kh")
    kw = tvm.reduce_axis((0, KW), name="kw")

    def _sparse_conv(n, nb, bs_r, h, w, vh, vw, vc):
        row_start = kernel_indptr[nb*VC+vc]
        row_end = kernel_indptr[nb*VC+vc + 1]
        row_elems = row_end - row_start

        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")

        block_offset = row_start + elem_idx

        return tvm.sum(data_vec[n][h][w][BI*kernel_indices[block_offset]+bi][vh*HSTR+kh][vw*WSTR+kw] *
                       kernel_data_vec[block_offset][bs_r][bi][kh][kw],
                       axis=[elem_idx, bi, kh, kw])

    ovshape = (N, NUM_BLKS // VC, BS_R, OH // VH, OW // VW, VH, VW, VC)
    conv = tvm.compute(ovshape, _sparse_conv, name="conv")

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod

    oshape = (N, CO, OH, OW)
    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n,
                              idxdiv(co, BS_R*VC), idxmod(co, BS_R), idxdiv(h, VH), idxdiv(w, VW),
                              idxmod(h, VH), idxmod(w, VW), idxdiv(idxmod(co, BS_R*VC), BS_R)],
                         name='output_unpack', tag='sparse_conv2d_output')

    return output

def schedule_sparse_conv2d_nchw_autotvm(cfg, s, data_vec, kernel_data_vec, conv, output, last):
    """schedule implementation"""
    #n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    n, nb, bs_r, oh, ow, vh, vw, vc = s[conv].op.axis
    elem_idx, bi, kh, kw = s[conv].op.reduce_axis

    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, nb, bs_r, oh, ow, bi, kh, kw, vh, vw, vc])
    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[cfg['tile_oh'].size[1],
                                        cfg['tile_ow'].size[1],
                                        cfg['tile_co'].size[1]],
                             max_unroll=16,
                             cfg=cfg)

    # schedule fusion
    n, co, h, w = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
        '''
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[1],
                                            cfg['tile_ow'].size[1],
                                            cfg['tile_co'].size[1]],
                                 max_unroll=16,
                                 cfg=cfg)
        '''
    s[conv].compute_at(s[last], ow)

    # mark parallel
    s[last].parallel(co)

    _, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    nnz, _, _, _, _ = s[kernel_data_vec].op.axis
    if autotvm.GLOBAL_SCOPE.in_tuning:
        s[kernel_data_vec].pragma(nnz, 'debug_skip_region')
    else:
        s[kernel_data_vec].parallel(nnz)
    #s[kernel_data_vec].parallel(nnz)

    return s
