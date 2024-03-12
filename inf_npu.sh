#!/usr/bin/bash
python inference.py --model_path models/yolov5s_relu.rknn --img_path models/cats.jpg --out_path models/cats-rknn.jpg --mode npu
