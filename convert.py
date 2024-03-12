import os, argparse
from rknn.api import RKNN


def main():

    parser = argparse.ArgumentParser('Convert ONNX model to RKNN runtime format, must run on x86_64')
    parser.add_argument('-i', '--inp_onnx', type=str, required=True, help='Input ONNX model')
    parser.add_argument('-o', '--out_rknn', type=str, required=True, help='Output RKNN model')
    parser.add_argument('-f', '--dataset_file', type=str, default='models/dataset.txt')

    args = parser.parse_args()

    rknn = RKNN(verbose=True)
    
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588')

    if rknn.load_onnx(model=args.inp_onnx) != 0:
        raise ValueError(f'Cannot load {args.inp_onnx}')

    if rknn.build(do_quantization=True, dataset=args.dataset_file) != 0:
        raise ValueError(f'Cannot build {args.inp_onnx}')

    if rknn.export_rknn(args.out_rknn) != 0:
        raise ValueError(f'Cannot export to {args.out_krnn}')


if __name__ == '__main__':
    main()
