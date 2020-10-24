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
 * \file src/relay/op/nn/tunable_sparse.h
 * \brief Properties def of tunable sparse operator for sharing.
 */
#ifndef TVM_RELAY_OP_NN_TUNABLE_SPARSE_H_
#define TVM_RELAY_OP_NN_TUNABLE_SPARSE_H_

#include <tvm/ir_pass.h>
#include <string>
#include <utility>

namespace tvm {
namespace relay {

template <typename AttrType>
bool SparseConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight_data = types[1].as<TensorTypeNode>();
  const auto* weight_indices = types[2].as<TensorTypeNode>();
  const auto* weight_indptr = types[3].as<TensorTypeNode>();

  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const AttrType* param = attrs.as<AttrType>();
  CHECK(param != nullptr);

  // Must define kernel size
  if (!param->kernel_size.defined()) return false;
  CHECK_EQ(param->kernel_size.size(), 2);

  // Sparse Conv2d only supports groups=1
  CHECK_EQ(param->groups, 1);

  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIHW);
  CHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCHW);
  CHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // Infer weight if the block_size and num_nnz_blocks and channels are defined
  if (param->block_size.defined() && param->num_nnz_blocks.defined() && param->channels.defined()) {
    CHECK_EQ(param->block_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wdata_shape;
    Array<IndexExpr> windices_shape;
    Array<IndexExpr> windptr_shape;

    wdata_shape = {{param->num_nnz_blocks, param->block_size[0],
                    param->block_size[1] * param->kernel_size[0] * param->kernel_size[1]}};
    windices_shape = {{param->num_nnz_blocks}};
    windptr_shape = {{indexdiv(param->channels, param->block_size[0]) + 1}};

    // TODO: Backward shape abuot weight data

    channels = param->channels;
    DataType weight_dtype = data->dtype;
    if (weight_data != nullptr) {
        weight_dtype = weight_data->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorTypeNode::make(wdata_shape, weight_dtype));
    reporter->Assign(types[2], TensorTypeNode::make(windices_shape, Int(32)));
    reporter->Assign(types[3], TensorTypeNode::make(windptr_shape, Int(32)));
  } else {
    // use weight to infer the conv shape.
    if (weight_data == nullptr || weight_indices == nullptr || weight_indptr == nullptr) return false;
    auto wdata_shape = weight_data->shape;
    auto windices_shape = weight_indices->shape;
    auto windptr_shape = weight_indptr->shape;
    CHECK_EQ(wdata_shape.size(), 3);
    CHECK_EQ(windices_shape.size(), 1);
    CHECK_EQ(windptr_shape.size(), 1);
    if (param->block_size.defined()) {
      CHECK_EQ(param->block_size.size(), 2);
      CHECK(reporter->AssertEQ(param->block_size[0], wdata_shape[1]) &&
            reporter->AssertEQ(param->block_size[1]*param->kernel_size[0]*param->kernel_size[1],
                               wdata_shape[2]))
          << "SparseConv2D: shape of weight_data is inconsistent with block_size, "
          << " block_size=" << param->block_size
          << " wdata_shape=" << wdata_shape;
      if (param->channels.defined()) {
        CHECK(reporter->AssertEQ(indexdiv(param->channels, param->block_size[0]) + 1,
                                 windptr_shape[0]))
            << "Sparseconv2D: shape of weight_indptr is inconsistent with block_size and channels, "
            << " block_size=" << param->block_size
            << " channels=" << param->channels
            << " windptr_shape=" << windptr_shape;
      }
    }
    if (param->num_nnz_blocks.defined()) {
      CHECK(reporter->AssertEQ(param->num_nnz_blocks, wdata_shape[0]))
          << "SparseConv2D: shape of weight_data is inconsistent with num_nnz_blocks, "
          << " num_nnz_blocks=" << param->num_nnz_blocks
          << " wdata_shape=" << wdata_shape;
      CHECK(reporter->AssertEQ(param->num_nnz_blocks, windices_shape[0]))
          << "SparseConv2D: shape of weight_indices is inconsistent with num_nnz_blocks, "
          << " num_nnz_blocks=" << param->num_nnz_blocks
          << " windices_shape=" << windices_shape;
    }
    channels = (windptr_shape[0] - 1) * wdata_shape[1];
  }
  dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
  dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];

  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  //GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (param->padding.size() == 1) {
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[0] * 2;
  } else if (param->padding.size() == 2) {
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[1] * 2;
  } else if (param->padding.size() == 4) {
    pad_h = param->padding[0] + param->padding[2];
    pad_w = param->padding[1] + param->padding[3];
  } else {
    CHECK_EQ(param->padding.size(), 4) << " Padding size should be 1, 2 or 4, but got "
        << param->padding.size();
  }

  if (!dshape_nchw[2].as<ir::Any>()) {
    oshape.Set(2, indexdiv(dshape_nchw[2] + pad_h - dilated_ksize_y,
                           param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_nchw[2]);
  }

  if (!dshape_nchw[3].as<ir::Any>()) {
    oshape.Set(3, indexdiv(dshape_nchw[3] + pad_w - dilated_ksize_x,
                           param->strides[1]) + 1);
  } else {
    oshape.Set(3, dshape_nchw[3]);
  }
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[4], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_TUNABLE_SPARSE_H_
