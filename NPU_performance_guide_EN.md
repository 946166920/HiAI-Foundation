# HiAI Foundation In-Depth User Guide
[TOC]

## NPU IR Operator Performance Guide

​		HiAI Foundation is dedicated to "AI on any device". In terms of usability, the provided operator functions are not limited. This guide introduces how to maximize IR performance based on the operator implementation mode.

### NN Operators

| IR operator | Optimal performance guide | Recommendation level                           |
|----------|----------|----------|
| Activation | The current hardware performance is optimal.   | ☆☆☆☆☆  |
| HardSwish | The current hardware performance is optimal. | ☆☆☆☆☆ |
| PRelu | The current hardware performance is optimal. | ☆☆☆☆☆ |
| BNInference | The current hardware performance is optimal. <br/>When Conv (depthwise) and Bn are used together, image fusion is performed.                                                   | ☆☆☆☆☆        |
| Convolution                       | (1) When Cin and Cout are both multiples of 16, the performance is optimal, reaching the maximum NPU computing power. <br/>(2) When both Cin and Cout are less than 16, the hardware computing power is:<br/>(Cin*Cout) / 256 * max computer power | ☆☆☆☆☆        |
| QuantizedConvolution | (1) When Cin and Cout are multiples of 32, the performance is optimal, reaching the maximum NPU computing power. <br/>(2) When both Cin and Cout are less than 32, the hardware computing power is:<br/>(Cin*Cout) / 1024 * max computer power | ☆☆☆☆☆        |
| ConvTranspose                     | (1) When Cin and Cout are both multiples of 16, the performance is optimal, reaching the maximum NPU computing power. <br/>(2) When both Cin and Cout are less than 16, the hardware computing power is<br/>(Cin*Cout) / 256 * max computer power<br/>(3) This optimization provides the optimal performance for kernel 1x1, 2x2, 3x3, and 8x8.  | ☆☆☆☆☆ |
| BiasAdd                           | The current hardware performance is optimal. <br/>When Conv (depthwise) and BiasAdd are used together, image fusion is performed.  | ☆☆☆☆☆ |
| Eltwise                           | The current hardware performance is optimal. | ☆☆☆☆☆ |
| LRN                               | The current hardware performance is good.<br>(1) The average variance is calculated during calculation, and the calculation amount is large. Therefore, the performance is worse than that of batchNorm. <br/>(2) Mainly used for image enhancement and is sensitive to precision calculation. If the NPU uses FP16 for calculation, there may be precision risks.  | ☆☆☆ |
| ConvolutionDepthwise              | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| QuantizedConvolutionDepthwise     | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| FullyConnection                   | The performance is limited by the DDR bandwidth. The operator is not limited by the computing power. Set the weight properly during algorithm design.  | ☆☆☆☆☆ |
| QuantizedFullyConnection          | The performance is limited by the DDR bandwidth. The operator is not limited by the computing power. Set the weight properly during algorithm design.  | ☆☆☆☆☆ |
| PoolingD                          | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| FractionalPooling                 | The function works, but the performance is poor. The operator contains a random number generator, instead of regular vector running.  | ☆ |
| Scale                             | The current hardware performance is optimal. <br/>Conv (depthwise) and Scale are used together, image fusion is performed.  | ☆☆☆☆☆ |
| ShuffleChannel                    | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆ |
| ShuffleChannelV2                  | Used for supporting operators in the ANN scenario. The function works, but the performance is poor.  | ☆ |
| Softmax                           | The current hardware performance is optimal. <br/>The performance is optimal for 4-dimensional input, when axis = 1, and softmax is performed based on the C channel.  | ☆☆☆☆☆ |
| TopK                              | Used for supporting operators in the ANN scenario. The function works, but the performance is poor.  | ☆ |
| LogSoftmax                        | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| Rank                              | Shape derivation operator, which can be offset during model building.  | ☆☆☆☆☆ |
| ScatterNd                         | Irregular data migration. The performance is poor, and not recommended for frequent use.  | ☆☆☆ |
| FakeQuantWithMinMaxVarsPerChannel | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| StopGradient                      | Shape derivation operator, which can be offset during model building.  | ☆☆☆☆☆ |
| LogicalXor                        | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| Threshold                         | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| AxisAlignedBboxTransform          | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| Clipboxes                         | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| Normalize                         | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| DecodeBBox                        | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| PSROIPooling                      | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| SVDF                              | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| ReduceMean                        | The current hardware performance is optimal.  | ☆☆☆☆☆ |
| LayerNorm                         | The current hardware performance is good.<br>(1) The average variance is calculated during calculation, and the calculation amount is large. Therefore, the performance is worse than that of batchNorm. <br/>(2) Mainly used for image enhancement and is sensitive to precision calculation. If the NPU uses FP16 for calculation, there may be precision risks.  | ☆☆☆          |
| InstanceNorm                      | The current hardware performance is good.<br>(1) The average variance is calculated during calculation, and the calculation amount is large. Therefore, the performance is worse than that of batchNorm. <br/>(2) Mainly used for image enhancement and is sensitive to precision calculation. If the NPU uses FP16 for calculation, there may be precision risks.  | ☆☆☆ |
| ROIPooling                        | The current hardware performance is optimal.                                            | ☆☆☆☆☆        |
| PriorBox                          | The current hardware performance is optimal.                                            | ☆☆☆☆☆        |
| LSTM                              | The performance is poor due to unordered data rearrangement. <br/>For details about the usage restrictions, see the IR comments and error logs generated during the actual building process. | ☆☆☆☆ |
| BidirectionLSTM                   | The performance is poor due to unordered data rearrangement. <br/>For details about the usage restrictions, see the IR comments and error logs generated during the actual building process. | ☆☆☆☆ |



### Math Operators
| IR operator            | Optimal performance guide                                 | Recommendation level |
| ----------------- | ------------------------------------------------ | ------------ |
| Add               | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Mul               | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Expm1             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Ceil              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Sin               | The performance is poor.                                         | ☆            |
| Cos               | The performance is poor.                                         | ☆            |
| Floor             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Log1p             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| LogicalAnd        | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| LogicalNot        | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Maximum           | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Minimum           | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Acosh             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Asinh             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Equal             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Reciprocal        | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Sqrt              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Square            | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| ReduceAllD        | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| CastT             | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Sign              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Cosh              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Exp               | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| FloorMod          | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| GreaterEqual      | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Greater           | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Less              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| MatMul            | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| RealDiv           | The performance is poor. It is recommended to use mul or Reciprocal+mul.        | ☆            |
| Rint              | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Round             | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Rsqrt             | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Sinh              | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| Sub               | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Range             | Optimized during model building.                              | ☆☆☆☆☆        |
| Acos              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Asin              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Atanh             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Log               | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| LogicalOr         | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Neg               | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| ReduceProdD       | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| ReduceSum         | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Tan               | The performance is poor.                                         | ☆            |
| Power             | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Pow               | The performance is poor.                                         | ☆            |
| ArgMaxExt2        | The current hardware performance is optimal.                                | ☆☆☆☆         |
| FloorDiv          | The performance is poor. Not recommended.                             | ☆            |
| NotEqual          | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| LessEqual         | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| ChannelAxpy       | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| SquaredDifference | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| Atan              | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| SegmentMax        | Unordered data reassembling and computing, resulting in poor performance.                       | ☆            |
| SegmentMin        | Unordered data reassembling and computing, resulting in poor performance.                       | ☆            |
| SegmentMean       | Unordered data reassembling and computing, resulting in poor performance.                       | ☆            |
| SegmentSum        | Unordered data reassembling and computing, resulting in poor performance.                       | ☆            |
| SegmentProd       | Unordered data reassembling and computing, resulting in poor performance.                       | ☆            |
| BatchMatMul       | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| ClipByValue       | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| L2Normalize       | The current hardware performance is optimal.                                | ☆☆☆☆☆        |
| ReduceMax         | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |
| ReduceMin         | The performance is better on phones using Kirin 9000 chips than those using other chips, on which the function can merely work.  | ☆            |



### Array Operators

| IR operator                  | Optimal performance guide                                             | Recommendation level |
| ----------------------- | ------------------------------------------------------------ | ------------ |
| ConcatD                 | The current hardware performance is optimal. <br/>When Cin and Cout are both multiples of 16, image fusion is performed to achieve the optimal performance.  | ☆☆☆☆☆        |
| FakeQuantWithMinMaxVars | The current hardware performance is optimal.                                            | ☆☆☆☆☆        |
| Reshape                 | The current hardware performance is optimal. <br/>In some scenarios, operators are fused and offset.           | ☆☆☆☆☆        |
| SplitD                  | The current hardware performance is optimal. <br/>When Cin and Cout are both multiples of 16, image fusion is performed to achieve the optimal performance.  | ☆☆☆☆☆        |
| SplitV                  | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Unpack                  | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Flatten                 | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Slice                   | The performance is poor due to unordered data rearrangement.                              | ☆            |
| ExpandDims              | Shape derivation operator, which can be offset during model building.                         | ☆☆☆☆☆        |
| GatherV2D               | The performance is poor due to unordered data rearrangement.                              | ☆            |
| GatherNd                | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Pack                    | The performance is poor due to unordered data rearrangement.                              | ☆            |
| SpaceToDepth            | The performance is poor due to unordered data rearrangement.                              | ☆            |
| DepthToSpace            | The performance is poor in most cases due to unordered data rearrangement. <br/>The performance is specially optimized and better in the four-grid scenario (Cin = 4, block = 1). | ☆☆           |
| StridedSlice            | The performance is poor due to unordered data rearrangement.                              | ☆            |
| SpaceToBatchND          | The performance is poor due to unordered data rearrangement.                              | ☆            |
| BatchToSpaceND          | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Tile                    | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Size                    | Shape derivation operator, which can be offset during model building.                         | ☆☆☆☆☆        |
| Fill                    | The performance is poor due to unordered data rearrangement.                              | ☆            |
| InvertPermutation       | The performance is poor due to unordered data rearrangement.                              | ☆            |
| Select                  | The function can merely work.                                                  | ☆☆           |
| ReverseSequence         | The performance is poor due to unordered data rearrangement.                              | ☆            |
| PadV2                   | The performance is optimal in the scenario where zeros are padded in the height and width directions. <br/>The performance is poor in other scenarios due to unordered data rearrangement.  | ☆☆☆          |
| Squeeze                 | Shape derivation operator, which can be offset during model building.                         | ☆☆☆☆☆        |
| BatchReindex            | The current hardware performance is optimal.                                            | ☆☆☆☆☆        |
| Pad                     | The performance is optimal in the scenario where zeros are padded in the height and width directions. <br/>The performance is poor in other scenarios due to unordered data rearrangement.  | ☆☆☆          |
| MirrorPad               | The performance is optimal in the scenario where zeros are padded in the height and width directions. The performance is poor in other scenarios due to unordered data rearrangement.                        | ☆            |
| OneHot                  | The performance is optimal in the scenario where zeros are padded in the height and width directions. The performance is poor in other scenarios due to unordered data rearrangement.                        | ☆            |
| Shape                   | Shape derivation operator, which can be offset during model building.                         | ☆☆☆☆☆        |
| Dequantize              | The current hardware performance is optimal.                                            | ☆☆☆☆☆        |
| Quantize                | The current hardware performance is optimal.                                            | ☆☆☆☆☆        |

### Detection Operators

| IR operator                  | Optimal performance guide                                             | Recommendation level |
| ----------------------- | ------------------------------------------------------------ | ------------ |
| Permute                 | The hardware is not suitable for too many such operations due to unordered data rearrangement, although related optimizations have been made.  | ☆☆☆          |
| FSRDetectionOutput      | The current performance is optimal.                                                | ☆☆☆☆☆        |
| DetectionPostprocessing | The current performance is optimal.                                                | ☆☆☆☆☆        |
| SSDDetectionOutput      | The current performance is optimal.                                                | ☆☆☆☆☆        |

### Image Operators

| IR operator | Optimal performance guide | Recommendation level |
| ------------------------------------------------------------ | ---------------------------------------- | ------------ |
| ImageData<br/>DynamicImageData<br/>ImageCrop<br/>ImageChannelSwap<br/>ImageColorSpaceConvertion<br/>ImageResize<br/>ImageDataTypeConversion<br/>ImagePadding | Operators related to AIPP image processing. The hardware performance is optimal.      | ☆☆☆☆☆        |
| CropAndResize                                                | The function works, but the performance is poor.                    | ☆            |
| ResizeBilinear<br/>ResizeBilinearV2<br/>Interp               | The hardware performance is optimal in most scenarios, but needs to be optimized in a few scenarios.  | ☆☆☆☆☆        |
| ResizeNearestNeighbor<br/>Upsample                           | The hardware performance is optimal in most scenarios, but needs to be optimized in a few scenarios.  | ☆☆☆☆☆        |
| Crop                                                         | The function works, but the performance is poor.                    | ☆            |
| NonMaxSuppressionV3D                                         | The function works, but the performance is poor.                    | ☆            |



## NPU Performance-Friendly Computing Structure

The current NPU-friendly structure and recommendation level are as follows:

> The following recommendation is provided after running the model after OMG conversion. In the IR interconnection scenario, ensure that there is no redundant reshape or permute operator.
>
> Note: The evaluation standards of recommendation are obtained based on the internal hardware usage of the NPU. However, other computing architectures (CPU and GPU) may also have similar problems in the unrecommended networks. The recommendations below are only for self-comparison, instead of horizontal comparisons.

| Application scenario | Network type                        | Recommendation level | Recommendation description                                                     |
| -------- | ------------------------------- | -------- | ------------------------------------------------------------ |
| Classification network | AlexNet                         | ☆☆☆☆     | The weight of the fully connected layer is large, and the bandwidth is limited during inference. Can be downloaded from the model zoo.                         |
|          | VGG16                           | ☆☆☆☆     | The weight of the fully connected layer is large, and the bandwidth is limited during inference. Can be downloaded from the model zoo.                          |
|          | VGG19                           | ☆☆☆      | The weight of the fully connected layer is large, and the bandwidth is limited during inference. Can be downloaded from the model zoo.                           |
|          | ResNet18/34/50/101/152          | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 100%. ResNet50 can be downloaded from the model zoo.                     |
|          | GoogleNet                       | ☆☆☆☆     | The hardware computing power usage is close to 75%. Can be downloaded from the model zoo.                                         |
|          | InceptionV3                     | ☆☆☆☆     | The hardware computing power usage is close to 85%. Can be downloaded from the model zoo.                                         |
|          | InceptionV4                     | ☆☆☆☆     | The hardware computing power usage is close to 85%. Can be downloaded from the model zoo.                                         |
|          | Inception_Resnet_v2             | ☆☆☆☆     | The hardware computing power usage is close to 90%. Can be downloaded from the model zoo.                                          |
|          | Xception                        | ☆☆☆☆     | The hardware computing power usage is close to 85%. Can be downloaded from the model zoo.                                        |
|          | MobileNet_v1                    | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 95%. Can be downloaded from the model zoo.                        |
|          | MobileNet_v2                    | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 95%. Can be downloaded from the model zoo.                      |
|          | MobileNet_v3                    | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 95%. Can be downloaded from the model zoo.                       |
|          | SqueezeNet                      | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 95%. Can be downloaded from the model zoo.                       |
|          | DenseNet                        | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 95%.                      |
|          | ShuffleNet_v1<br/>ShuffleNet_v2 | ☆        | A large number of shuffleChannel operations are involved. These operations are memory migration operations and are not computing-limited. <br/>The bandwidth of the network is limited. The shuffleChannel supports the function but does not ensure the optimal performance.  |
|          | Resnext                         | ☆☆☆☆     | The hardware computing power usage is close to 85%.                                        |
|          | EfficientNet                    | ☆☆☆☆☆    | The weight of the model is moderate, and the hardware computing power usage is close to 95%.                      |
|          | SENet                           | ☆☆☆☆     | The hardware computing power usage is close to 75%.                                        |
| Object detection | Faster_RCNN                     | ☆☆☆☆☆    | The hardware computing power usage is close to 85%. |
|          | SSD                             | ☆☆☆☆     | The hardware computing power usage is close to 85%. Can be generated only through the OMG process.            |
|          | FPN                             | ☆☆☆☆☆    | The hardware computing power usage is close to 90%. The post-processing is not included in the model, but is implemented by the algorithm independently.     |
| Voice segmentation | FCN                             | ☆☆☆☆☆    | The hardware computing power usage is close to 85%. Because the model computing volume is large, parameters need to be tailored during actual deployment. Can be downloaded from the model zoo.   |
|          | DeeplabV3                       | ☆☆☆      | The hardware computing power usage is close to 60%. Can be downloaded from the model zoo.                                       |
|          | Unet                            | ☆☆☆      | The hardware computing power usage is close to 60%.                                       |
|          | MarskRcnn                       | ☆☆       | The hardware computing power usage is close to 80% (only for the TF-to-OM version, not supported in IR interconnection mode).   |
|          | PSPNet                          | ☆☆☆      | The pyramid pooling operators are not supported. Can be equivalent to multiple pools, but the performance is average.   |
| Super resolution | VDSR                            | ☆☆☆☆☆    | The hardware computing power usage is close to 85%. Meets the real-time super-resolution requirements. Can be downloaded from the model zoo.                  |
|          | FSRCNN                          | ☆☆☆☆     | The hardware computing power usage is close to 70%. Meets partial real-time super-resolution requirements. Can be downloaded from the model zoo.              |
|          | SRCNN                           | ☆☆☆☆     | The hardware computing power usage is close to 70%. Meets partial real-time super-resolution requirements.             |
|          | DnCNN                           | ☆☆☆☆     | The hardware computing power usage is close to 65%. Due to large computing volume, only partial real-time super-resolution requirements can be met.  |
|          | DRCN                            | ☆☆☆☆     | The hardware computing power usage is close to 65%. Due to large computing volume, only partial real-time super-resolution requirements can be met.  |
|          | DRRN                            | ☆☆☆      | The hardware computing power usage is close to 60%. Due to large computing volume, only partial real-time super-resolution requirements can be met.  |
|          | EnhanceNet                      | ☆☆☆      | The hardware computing power usage is close to 60%. Due to large computing volume, only partial real-time super-resolution requirements can be met.  |
| Voice semantics | RNN                             | ☆☆       | The supported functions are limited.                                  |
|          | LSTM                            | ☆☆       | The supported functions are limited.                                  |
|          | Transformer                     | ☆☆☆☆     | The hardware computing power usage is close to 70%.             |
|          | Bert                            | ☆☆☆☆     | The hardware computing power usage is close to 70%.             |
