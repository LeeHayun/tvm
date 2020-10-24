/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tunable_sparse.cc
 * \brief Tunable sparse operators
 */
#include <tvm/data_layout.h>
#include <tvm/ir_pass.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../pass/alter_op_layout.h"
#include "tunable_sparse.h"

namespace tvm {
namespace relay {

template<typename T>
Array<Array<Layout > > SparseConvInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>> &old_in_shapes) {
  const T* params = attrs.as<T>();

  return Array<Array<Layout> >{{params->data_layout, "BRC", "B", "P"},
                               {params->out_layout == "" ?
                                    params->data_layout : params->out_layout}};
}

// relay.nn.sparse_conv2d
TVM_REGISTER_NODE_TYPE(SparseConv2DAttrs);


// Positional relay function to create conv2d operator
// used by frontend FFI.
Expr MakeSparseConv2D(Expr data,
                Expr weight_data,
                Expr weight_indices,
                Expr weight_indptr,
                Array<IndexExpr> block_size,
                IndexExpr num_nnz_blocks,
                Array<IndexExpr> strides,
                Array<IndexExpr> padding,
                Array<IndexExpr> dilation,
                int groups,
                IndexExpr channels,
                Array<IndexExpr> kernel_size,
                std::string data_layout,
                std::string kernel_layout,
                std::string out_layout,
                DataType out_dtype) {
  auto attrs = make_object<SparseConv2DAttrs>();
  attrs->block_size = std::move(block_size);
  attrs->num_nnz_blocks = std::move(num_nnz_blocks);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.sparse_conv2d");
  return CallNode::make(op, {data, weight_data, weight_indices, weight_indptr}, Attrs(attrs), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.sparse_conv2d")
.set_body_typed(MakeSparseConv2D);


RELAY_REGISTER_OP("nn.sparse_conv2d")
.describe(R"code(Sparse 2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight_data**: (num_nnz_blocks, block_size[0], block_size[1] * kernel_size[0] * kernel_size[1])
- **weight_indices**: (num_nnz_blocks) 
- **weight_indptr**: (out_channels // block_size[0] + 1)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type<SparseConv2DAttrs>()
.set_num_inputs(4)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight_data", "1D Tensor", "Weight tensor matrix.")
.add_argument("weight_indices", "1D Tensor", "Weight indices matrix.")
.add_argument("weight_indptr", "1D Tensor", "Weight indptr matrix.")
.set_support_level(2)
.add_type_rel("SparseConv2D", SparseConv2DRel<SparseConv2DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", SparseConvInferCorrectLayout<SparseConv2DAttrs>);


}  // namespace relay
}  // namespace tvm
