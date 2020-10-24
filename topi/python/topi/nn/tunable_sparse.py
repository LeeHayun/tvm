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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""Tunable sparse operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm

from .pad import pad
from .util import get_pad_tuple

@tvm.target.generic_func
def sparse_conv2d(input, filter_data, filter_indices, filter_indptr, kernel_size, strides, padding, dilation, layout='NCHW', out_dtype=None):
    # search platform specific declaration first
    # default declaration
    if layout == 'NCHW':
        return sparse_conv2d_nchw(input, filter_data, filter_indices, filter_indptr, kernel_size, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))

def sparse_conv2d_nchw(Input, Filter_data, Filter_indices, Filter_indptr, kernel_size, stride, padding, dilation, out_dtype=None):
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(kernel_size, int) or len(kernel_size) == 2
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    if len(Filter_data.shape) == 1:
        num_filter = Filter_indptr.shape[0] - 1
    elif len(Filter_data.shape) == 3:
        num_filter = (Filter_indptr.shape[0] - 1) * Filter_data.shape[1]
    else:
        raise ValueError("Only CSR or BSR supported")
    channel = in_channel
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if pad_top != 0 or pad_left != 0 or pad_down != 0 or pad_right != 0:
        temp = pad(Input, pad_before, pad_after, name="pad_temp")
    else:
        temp = Input

    """
    '''
    im2col_out_shape = (batch, in_channel, kernel_h, kernel_w, out_height, out_width)
    def f(batch, c, h, w , out_h, out_w):
        return temp[batch, c, h + stride_h*out_h, w + stride_w*out_w]
    im2col_input = tvm.compute(im2col_out_shape, f, name='col')
    im2col_input = topi.transpose(im2col_input, axes=[0, 4, 5, 1, 2, 3])
    im2col_input = topi.reshape(im2col_input, newshape = (batch*out_height*out_width, in_channel*kernel_h*kernel_w))
    '''
    im2col_input = tvm.compute([batch * out_height * out_width, in_channel * kernel_h * kernel_w], lambda i, j:
            temp[i // (out_height*out_width)][j // (kernel_h*kernel_w)][i // out_width % out_height*stride_h + j // kernel_w % kernel_h][i % out_width*stride_w + j % kernel_w], name='col')
    '''
    im2col_input = tvm.compute((batch, out_height, out_width, in_channel, kernel_h, kernel_w), lambda n, oh, ow, ci, hk, wk: \
            temp[n][ci][oh*stride_h+hk][ow*stride_w+wk], name='col')
    im2col_input = topi.reshape(im2col_input, newshape=(batch*out_height*out_width, in_channel*kernel_h*kernel_w))
    '''

    output = topi.nn.sparse_dense(im2col_input, Filter_data, Filter_indices, Filter_indptr)
    output = topi.transpose(output, axes=[1, 0])
    output_shape = (batch, out_channel, out_height, out_width)
    output = topi.reshape(output, newshape=output_shape)

    '''
    s = tvm.create_schedule(output.op)
    print(tvm.lower(s, [Input, Filter_data, Filter_indices, Filter_indptr, output], simple_mode=True))
    '''
    """

    im2col_out_shape = (batch, in_channel, kernel_h, kernel_w, out_height, out_width)
    def f(b, c, h, w , out_h, out_w):
        return temp[b, c, h + stride_h*out_h, w + stride_w*out_w]
    im2col_input = tvm.compute(im2col_out_shape, f, name='col')
    im2col_input = topi.reshape(im2col_input, newshape=(batch*in_channel*kernel_h*kernel_w, out_height*out_width))

    output = _sparse_dense_bsrmm_v2(im2col_input, Filter_data, Filter_indices, Filter_indptr)
    output_shape = (batch, out_channel, out_height, out_width)
    output = topi.reshape(output, newshape=output_shape)

    return output
