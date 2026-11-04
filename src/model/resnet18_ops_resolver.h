#ifndef RESNET18_OPS_RESOLVER_H_
#define RESNET18_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/c/common.h"

using ResNet18OpResolver = tflite::MicroMutableOpResolver<7>;

inline TfLiteStatus RegisterResNet18Ops(ResNet18OpResolver& resolver) {
  TF_LITE_ENSURE_STATUS(resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(resolver.AddPad());
  TF_LITE_ENSURE_STATUS(resolver.AddTranspose());
  return kTfLiteOk;
}

#endif  // RESNET18_OPS_RESOLVER_H_
