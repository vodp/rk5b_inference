#!/usr/bin/bash
python inference.py --model_path models/yolov5s_relu.onnx --img_path models/cats.jpg --out_path models/cats-onnx.jpg --mode onnx
