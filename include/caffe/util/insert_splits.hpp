#ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
#define _CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
CAFFE_FUNCTION_EXPORTS void InsertSplits(const NetParameter& param, NetParameter* param_split);

CAFFE_FUNCTION_EXPORTS void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param);

CAFFE_FUNCTION_EXPORTS string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx);

CAFFE_FUNCTION_EXPORTS string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
