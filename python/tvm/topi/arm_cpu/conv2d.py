# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, no-else-return, unused-argument, import-outside-toplevel
"""Conv2D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import autotvm
import tvm.contrib.nnpack

from ..util import traverse_inline, get_const_tuple
from .. import nn
from ..nn.util import get_const_int, get_pad_tuple
from ..nn.winograd_util import winograd_transform_matrices
from .conv2d_spatial_pack import (
    conv2d_spatial_pack_nchw,
    conv2d_spatial_pack_nhwc,
    schedule_conv2d_spatial_pack_nchw,
    schedule_conv2d_spatial_pack_nhwc,
)
from .cortex_m7.conv2d import direct_simd


@autotvm.register_topi_compute("conv2d_nchw_spatial_pack.arm_cpu")
def conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NCHW layout"""
    return conv2d_spatial_pack_nchw(
        cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=2
    )


@autotvm.register_topi_schedule("conv2d_nchw_spatial_pack.arm_cpu")
def schedule_conv2d_nchw_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nchw"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if "spatial_conv2d_output" in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_spatial_pack.arm_cpu")
def conv2d_nhwc_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NHWC layout"""
    return conv2d_spatial_pack_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_spatial_pack.arm_cpu")
def schedule_conv2d_nhwc_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nhwc"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "spatial_conv_output_NHWC" in op.tag:
            schedule_conv2d_spatial_pack_nhwc(cfg, s, op, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nchw_winograd.arm_cpu")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw layout using Winograd with weight transform"""
    tile_size = 4
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size)


@autotvm.register_topi_schedule("conv2d_nchw_winograd.arm_cpu")
def schedule_conv2d_nchw_winograd(cfg, outs):
    """Create schedule for conv2d_nchw_winograd"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_conv2d_output" in op.tag:
            output = op.output(0)
            _schedule_winograd(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size):
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")
    if not isinstance(IH, int) or not isinstance(IW, int):
        raise RuntimeError("ARM winograd conv2d doesn't support dynamic input height or width.")

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
    data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    K = CO
    C = CI

    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    # TODO(@kevinthesun): Support tuning/optimization for dynamic shape.
    tile_p = P if isinstance(N, int) else nH * nW
    cfg.define_split("tile_p", cfg.axis(tile_p), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    cfg.define_split("tile_k", cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = cfg["tile_p"].size[-1]
    VK = cfg["tile_k"].size[-1]

    # pack input tile
    input_tile = te.compute(
        (C, idxd(P, VP), alpha, alpha, VP),
        lambda c, b, eps, nu, bb: data_pad[
            idxd(b * VP + bb, nH * nW),
            c,
            idxm(idxd(b * VP + bb, nW), nH) * m + eps,
            idxm(b * VP + bb, nW) * m + nu,
        ],
        name="d",
    )

    if autotvm.GLOBAL_SCOPE.in_tuning:
        VC = cfg["tile_k"].size[-1]
        kvshape = (KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), CI, VC)
        U = tvm.te.placeholder(kvshape, kernel.dtype, name="U")
    else:
        # transform kernel
        if pre_computed:
            U = kernel
        else:
            r_kh = te.reduce_axis((0, KH), "r_kh")
            r_kw = te.reduce_axis((0, KW), "r_kw")
            U = te.compute(
                (alpha, alpha, idxd(K, VK), C, VK),
                lambda eps, nu, k, c, kk: te.sum(
                    kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype)
                    * G[eps][r_kh]
                    * G[nu][r_kw],
                    axis=[r_kh, r_kw],
                ),
                name="U",
            )

    # transform image
    r_eps = te.reduce_axis((0, alpha), "r_eps")
    r_nu = te.reduce_axis((0, alpha), "r_nu")
    V = te.compute(
        (alpha, alpha, idxd(P, VP), C, VP),
        lambda eps, nu, b, c, bb: te.sum(
            input_tile[c][b][r_eps][r_nu][bb].astype(out_dtype) * B[r_eps][eps] * B[r_nu][nu],
            axis=[r_eps, r_nu],
        ),
        name="V",
    )

    # batch gemm
    c = te.reduce_axis((0, C), name="c")
    M = te.compute(
        (alpha, alpha, K, P),
        lambda eps, nu, k, b: te.sum(
            U[eps][nu][idxd(k, VK)][c][idxm(k, VK)] * V[eps][nu][idxd(b, VP)][c][idxm(b, VP)],
            axis=c,
        ),
        name="M",
    )

    # inverse transform
    r_eps = te.reduce_axis((0, alpha), "r_eps")
    r_nu = te.reduce_axis((0, alpha), "r_nu")
    Y = te.compute(
        (K, P, m, m),
        lambda k, b, vh, vw: te.sum(
            M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw], axis=[r_eps, r_nu]
        ),
        name="Y",
    )

    # unpack output
    output = te.compute(
        (N, K, H, W),
        lambda n, k, h, w: Y[k][n * nH * nW + idxd(h, m) * nW + idxd(w, m), idxm(h, m), idxm(w, m)],
        name="output",
        tag="winograd_conv2d_output",
    )

    # we have to manually assign effective GFLOP for winograd
    if isinstance(N, int):
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output


def _schedule_winograd(cfg, s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    U, V = M.op.input_tensors
    d, B = V.op.input_tensors
    data_pad = d.op.input_tensors[0]

    # padding
    s[data_pad].compute_inline()

    # pack input tiles
    s[d].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.te.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        (
            eps,
            nu,
            k,
            c,
            kk,
        ) = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, "debug_skip_region")
        else:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
            for axis in [eps, nu, r_kh, r_kw]:
                s[U].unroll(axis)
            s[U].vectorize(kk)
            s[U].parallel(k)

        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    DD = s.cache_read(d, "global", [V])
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, eps, nu, r_eps, r_nu, bb)
    for axis in [eps, nu, r_eps, r_nu]:
        s[V].unroll(axis)
    s[DD].compute_at(s[V], c)
    s[V].vectorize(bb)
    s[V].parallel(b)

    # batch gemm
    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    cfg.define_split("tile_c", c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    co, ci = cfg["tile_c"].apply(s, M, c)
    xo, xi = cfg["tile_p"].apply(s, M, b)
    s[M].reorder(eps, nu, xo, co, k, ci, xi)
    cfg.define_annotate("ann_reduce", [ci], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [k, xi], policy="try_unroll_vec")
    cfg["ann_reduce"].apply(s, M, [ci], axis_lens=[cfg["tile_c"].size[-1]], max_unroll=16, cfg=cfg)
    cfg["ann_spatial"].apply(s, M, [k, xi])

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    for axis in [vh, vw, r_eps, r_nu]:
        s[Y].unroll(axis)

    # output
    n, co, h, w = s[last].op.axis
    co, coi = cfg["tile_k"].apply(s, last, co)
    p = s[last].fuse(n, co)
    s[M].compute_at(s[last], p)
    s[last].parallel(p)

    MM = s.cache_read(M, "global", [Y])
    m = get_const_int(V.shape[0]) + 1 - 3
    ho, wo, hi, wi = s[last].tile(h, w, m, m)
    s[Y].compute_at(s[last], wo)
    s[MM].compute_at(s[last], wo)

    if output != last:
        s[output].compute_inline()


@autotvm.register_topi_compute("conv2d_nchw_winograd_nnpack.arm_cpu")
def conv2d_nchw_winograd_nnpack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw using nnpack Winograd implementation"""
    dtype = data.dtype
    if dtype == "float32":
        return _conv2d_arm_cpu_winograd_nnpack(
            cfg,
            data,
            kernel,
            strides,
            padding,
            dilation,
            out_dtype,
            tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8,
        )
    elif dtype == "float16":
        return _conv2d_arm_cpu_winograd_nnpack(
            cfg,
            data,
            kernel,
            strides,
            padding,
            dilation,
            out_dtype,
            tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8_FP16,
        )
    else:
        raise ValueError("Unsupported data type {} for conv2d winograd nnpack".format(dtype))


@autotvm.register_topi_schedule("conv2d_nchw_winograd_nnpack.arm_cpu")
def schedule_conv2d_nchw_winograd_nnpack(cfg, outs):
    """Create schedule for conv2d_nchw_winograd_nnpack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_nnpack_conv2d_output" in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _conv2d_arm_cpu_winograd_nnpack(
    cfg, data, kernel, strides, padding, dilation, out_dtype, convolution_algorithm
):
    """ TOPI compute callback. Use winograd NNPACK template """
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(kernel.shape) == 4
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert (
        KH == 3
        and KW == 3
        and pt == 1
        and pb == 1
        and pl == 1
        and pr == 1
        and HSTR == 1
        and WSTR == 1
    )
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1

    cfg.define_knob("winograd_nnpack_algorithm", [convolution_algorithm])

    assert N == 1
    with tvm.te.tag_scope("winograd_nnpack_conv2d_weight_transform"):
        transformed_kernel = tvm.contrib.nnpack.convolution_inference_weight_transform(
            kernel, algorithm=cfg["winograd_nnpack_algorithm"].val
        )
        if autotvm.GLOBAL_SCOPE.in_tuning:
            transformed_kernel = te.compute(transformed_kernel.shape, lambda *args: 0.0)

    with tvm.te.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data,
            transformed_kernel,
            bias=None,
            padding=[pt, pb, pl, pr],
            stride=[HSTR, WSTR],
            algorithm=cfg["winograd_nnpack_algorithm"].val,
        )

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


def _schedule_winograd_nnpack(cfg, s, output, last):
    # Could have bias.

    (X, TK) = output.op.input_tensors[:2]

    # transform kernel
    assert isinstance(TK.op, (te.tensor.ComputeOp, te.tensor.ExternOp, te.tensor.PlaceholderOp))
    if autotvm.GLOBAL_SCOPE.in_tuning and isinstance(TK.op, te.tensor.ComputeOp):
        # kernel transformation will be pre-computed during compilation, so we skip
        # this part to make tuning records correct
        s[TK].pragma(s[TK].op.axis[0], "debug_skip_region")


@autotvm.register_topi_compute("conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
def conv2d_nchw_winograd_nnpack_without_weight_transform(
    cfg, data, transformed_kernel, bias, strides, padding, dilation, out_dtype
):
    """Compute conv2d_nchw using NNPack winograd without weight transform"""
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(transformed_kernel.shape) == 4
    CO, _, _, _ = get_const_tuple(transformed_kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    KH, KW = 3, 3
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert (
        KH == 3
        and KW == 3
        and pt == 1
        and pb == 1
        and pl == 1
        and pr == 1
        and HSTR == 1
        and WSTR == 1
    )
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1

    assert N == 1
    with tvm.te.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data=data,
            transformed_kernel=transformed_kernel,
            bias=bias,
            padding=[pt, pb, pl, pr],
            stride=[HSTR, WSTR],
            algorithm=cfg["winograd_nnpack_algorithm"].val,
        )

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


@autotvm.register_topi_schedule("conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
def schedule_conv2d_nchw_winograd_nnpack_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_nnpack_conv2d_output" in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_direct_simd.arm_cpu")
def conv2d_direct_simd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with SIMD (v7e-m)."""
    return direct_simd.conv2d_direct_simd_compute(
        cfg, data, kernel, strides, padding, dilation, out_dtype
    )


@autotvm.register_topi_schedule("conv2d_direct_simd.arm_cpu")
def schedule_conv2d_direct_simd(cfg, outs):
    """Create schedule for conv2d_direct_simd"""
    return direct_simd.conv2d_direct_simd_nhwc_schedule(cfg, outs)



@autotvm.register_topi_compute("sparse_conv2d_nchw.arm_cpu")
def sparse_conv2d_nchw(cfg, data, kernel_data, kernel_indices, kernel_indptr, kernel_size, strides, padding, dilation, layout, out_dtype):
    if layout == 'NCHW':
        if isinstance(kernel_size, int):
            KH = KW = kernel_size
        else:
            KH, KW = kernel_size

        if KH == 1 and KW == 1:
            return sparse_conv2d_nchw_autotvm(cfg, data, kernel_data, kernel_indices, kernel_indptr,
                                      kernel_size, strides, padding, dilation, out_dtype)
        else:
            return sparse_depthwise_conv2d_nchw_autotvm(cfg, data, kernel_data, kernel_indices, kernel_indptr,
                                                        kernel_size, strides, padding, dilation, out_dtype)
    else:
        raise ValueError("Unsupported layout {}.".format(layout))


@autotvm.register_topi_schedule("sparse_conv2d_nchw.arm_cpu")
def schedule_sparse_conv2d_nchw(cfg, outs):
    s = te.create_schedule([x.op for x in outs])
    conv = None
    data = None
    data_vec = None
    data_pad = None
    kernel_data_vec = None
    output = None
    kernel_data = None
    kernel_indptr = None
    kernel_indices = None

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

        if op.tag == 'spatial_depthwise_conv2d_nchw_output':
            output = op.output(0)
            conv = op.input_tensors[0]
            data_vec = conv.op.input_tensors[2]
            kernel_data_vec = conv.op.input_tensors[3]

            #data_pad = data_vec.op.input_tensors[0]
            #data = data_pad.op.input_tensors[0]
            #kernel_indptr = conv.op.input_tensors[0]
            #kernel_indices = conv.op.input_tensors[1]
            #kernel_data = kernel_data_vec.op.input_tensors[0]


            _schedule_spatial_pack(cfg, s, data_vec, kernel_data_vec, conv, output, outs[0])
            #print(tvm.lower(s, [data, kernel_data, kernel_indices, kernel_indptr], simple_mode=True))

    traverse_inline(s, outs[0].op, _callback)
    return s

def sparse_conv2d_nchw_autotvm(cfg, data, kernel_data, kernel_indices, kernel_indptr,
                               kernel_size, strides, padding, dilation, out_dtype):
    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)

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

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (KH, KW))
    OH = (IH + pad_top + pad_bottom - KH) // HSTR + 1
    OW = (IW + pad_left + pad_right - KW) // WSTR + 1
    data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])

    NNZ, BS_O, BS_I_KH_KW = get_const_tuple(kernel_data.shape)
    BS_I = BS_I_KH_KW // (KH * KW)
    NUM_BLKS_PLUS_1, = get_const_tuple(kernel_indptr.shape)
    NB_O = NUM_BLKS_PLUS_1 - 1    # NUM_BLKS = CO // BS_O
    NB_I = CI // BS_I


    nb_o, bs_o, oh, ow = cfg.axis(NB_O), cfg.axis(BS_O), cfg.axis(OH), cfg.axis(OW)
    kh, kw = cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    nb_o, vnb_o = cfg.define_split('tile_nb_o', nb_o, num_outputs=2, candidate=[[-1,4]])
    oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
    ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)

    cfg.define_reorder("reorder_0",
                       [vh, vw, vnb_o, bs_o],
                       policy='candidate', candidate=[
                           [vh, vw, vnb_o, bs_o],
                           [vnb_o, bs_o, vh, vw]])

    cfg.define_annotate('ann_reduce', [kh, kw], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [vh, vw, vnb_o, bs_o], policy='try_unroll_vec')

    VNB_O = cfg['tile_nb_o'].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (N, OH // VH, OW // VW, NB_I, BS_I, VH, VW)
    data_vec = te.compute(dvshape, lambda n, h, w, nb_i, bs_i, vh, vw:
                           data_pad[n][nb_i*BS_I+bs_i][h*VH*HSTR+vh][w*VW*WSTR+vw],
                           name='data_vec')

    kdvshape = (NNZ, BS_I, KH, KW, BS_O)
    kernel_data_vec = te.compute(kdvshape, lambda nnz, bs_i, kh, kw, bs_o:
                                  kernel_data[nnz, bs_o, bs_i*KH*KW+kh*KW+kw],
                                  name='kernel_data_vec')

    bs_i = te.reduce_axis((0, BS_I), name='bs_i')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    def _sparse_conv2d(n, nb_o, h, w, vh, vw, vnb_o, bs_o):
        row_start = kernel_indptr[nb_o*VNB_O+vnb_o]
        row_end = kernel_indptr[nb_o*VNB_O+vnb_o + 1]
        row_elems = row_end - row_start

        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx

        data_val = data_vec[n][h][w][kernel_indices[block_offset]][bs_i][vh*HSTR+kh][vw*WSTR+kw]
        kernel_val = kernel_data_vec[block_offset][bs_i][kh][kw][bs_o]

        return te.sum(data_val * kernel_val, axis=[elem_idx, bs_i, kh, kw])

    ovshape = (N, NB_O // VNB_O, OH // VH, OW // VW, VH, VW, VNB_O, BS_O)
    conv = te.compute(ovshape, _sparse_conv2d, name='conv')

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    oshape = (N, CO, OH, OW)
    output = te.compute(oshape, lambda n, co, h, w:
                         conv[n,
                              idxd(co, VNB_O*BS_O), idxd(h, VH), idxd(w, VW),
                              idxm(h, VH), idxm(w, VW), idxm(idxd(co, BS_O), VNB_O), idxm(co, BS_O)],
                         name='output_unpack', tag='sparse_conv2d_output')

    '''
    dvshape = (N, OH, OW, NB_I, BS_I)
    data_vec = tvm.compute(dvshape, lambda n, h, w, nb_i, bs_i:
                           data_pad[n][nb_i*BS_I+bs_i][h*HSTR][w*WSTR],
                           name='data_vec')

    kdvshape = (NNZ, KH, KW, BS_I, BS_O)
    kernel_data_vec = tvm.compute(kdvshape, lambda nnz, kh, kw, bs_i, bs_o:
                                  kernel_data[nnz][bs_o][bs_i*KH*KW + kh*KW + kw],
                                  name='kernel_data_vec')

    bs_i = tvm.reduce_axis((0, BS_I), name='bs_i')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    def _sparse_conv2d(n, h, w, nb_o, bs_o):
        row_start = kernel_indptr[nb_o]
        row_end = kernel_indptr[nb_o + 1]
        row_elems = row_end - row_start

        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx

        data_val = data_vec[n][h*HSTR+kh][w*WSTR+kw][kernel_indices[block_offset]][bs_i]
        kernel_val = kernel_data_vec[block_offset][kh][kw][bs_i][bs_o]

        return tvm.sum(data_val * kernel_val, axis=[elem_idx, kh, kw, bs_i])

    ovshape = (N, OH, OW, NB_O, BS_O)
    conv = tvm.compute(ovshape, _sparse_conv2d, name='conv')

    idxd = tvm.indexdiv
    idxm = tvm.indexmod

    oshape = (N, CO, OH, OW)
    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][h][w][idxd(co, BS_O)][idxm(co, BS_O)],
                         name='output',
                         tag='sparse_conv2d_output')
    '''

    # ================== define configuration space =========================
    cfg.add_flop(2*N*OH*OW*KH*KW*BS_O*BS_I*NNZ)


    return output

def schedule_sparse_conv2d_nchw_autotvm(cfg, s, data_vec, kernel_data_vec, conv, output, last):
    n, nb_o, oh, ow, vh, vw, vnb_o, bs_o = s[conv].op.axis
    elem_idx, bs_i, kh, kw = s[conv].op.reduce_axis
    BS_O = get_const_int(s[kernel_data_vec].op.axis[-1].dom.extent)

    # schedule conv
    s[conv].reorder(n, nb_o, oh, ow, elem_idx, bs_i, kh, kw, vh, vw, vnb_o, bs_o)
    cfg['reorder_0'].apply(s, conv, [vh, vw, vnb_o, bs_o])
    cfg['ann_reduce'].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg['ann_spatial'].apply(s, conv, [vh, vw, vnb_o, bs_o],
                             axis_lens=[cfg['tile_oh'].size[-1],
                                        cfg['tile_ow'].size[-1],
                                        cfg['tile_nb_o'].size[-1],
                                        BS_O],
                             max_unroll=16,
                             cfg=cfg)

    # scehdule fusion
    n, co, h, w = s[last].op.axis
    nb_o, bs_o = s[last].split(co, BS_O)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    nb_o, vnb_o = cfg['tile_nb_o'].apply(s, last, nb_o)
    s[last].reorder(n, nb_o, oh, ow, vh, vw, vnb_o, bs_o)
    if last != output:
        s[output].compute_inline()
        cfg['ann_spatial'].apply(s, last, [vh, vw, vnb_o, bs_o],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_nb_o'].size[-1],
                                            BS_O],
                                 max_unroll=16,
                                 cfg=cfg)
    #cfg['reorder_0'].apply(s, last, [vh, vw, vnb_o, bs_o])
    s[conv].compute_at(s[last], ow)

    # mark parallel
    s[last].parallel(nb_o)

    #_, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(s[data_vec].op.axis[1])

    #nnz, _, _, _, _ = s[kernel_data_vec].op.axis
    s[kernel_data_vec].parallel(s[kernel_data_vec].op.axis[0])

    return s

    '''
    num_spatial_axis = len(s[conv].op.axis)
    parallel_candidate = [i+1 for i in range(num_spatial_axis*2)]

    _define_split(s, cfg, conv, pattern='ssnrsrs')
    cfg.define_knob("auto_unroll_max_step", [0, 16, 64, 512])
    cfg.define_knob("follow_split", [1, 2])
    cfg.define_knob("parallel", parallel_candidate)


    """schedule implementation"""
    # schedule fusion
    _apply_and_reorder(s, cfg, conv, pattern='ssnrsrs')

    BS_O = get_const_int(s[data_vec].op.axis[-1].dom.extent)

    n, c, h , w = s[last].op.axis
    nbo, bs_o = s[last].split(c, BS_O)
    s[last].reorder(n, h, w, nbo, bs_o)
    _follow_split(s, cfg, last, conv, n_split=cfg['follow_split'].val)

    if last != output:
        s[output].compute_inline()

    _compute_at(s, conv, last, cfg['follow_split'].val * num_spatial_axis)

    # Inline Step
    #s[reshaped_data_pad].compute_inline()
    #s[data_pad].compute_inline()

    # vectorize
    #tmp = _parallel(s, conv, 5)
    _vectorize(s, conv)
    _vectorize(s, last)

    # mark parallel
    _parallel(s, data_vec, 2)
    tmp = _parallel(s, last, cfg['parallel'].val)
    s[last].pragma(tmp, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    nnz = s[kernel_data_vec].op.axis[0]
    if autotvm.GLOBAL_SCOPE.in_tuning:
        s[kernel_data_vec].pragma(nnz, 'debug_skip_region')
    else:
        s[kernel_data_vec].parallel(nnz)

    return s
    '''
