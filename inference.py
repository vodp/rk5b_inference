import os, argparse, cv2
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

from coco_utils import COCO_test_helper
from utils import post_process, draw, IMG_SIZE


type_map = {
    'tensor(int32)' : np.int32,
    'tensor(int64)' : np.int64,
    'tensor(float32)' : np.float32,
    'tensor(float64)' : np.float64,
    'tensor(float)' : np.float32,
}


class OnnxModelContainer:
    ''' A wrapper that maintains an active session attached to a model
    '''
    def __init__(self, model_path):

        options = ort.SessionOptions()
        options.log_severity_level = 2
        self.sess = ort.InferenceSession(
                        model_path, 
                        sess_options=options, 
                        providers=['CPUExecutionProvider'])
        self.model_path = model_path

    def run(self, inputs):
        ''' Check validity of inputs and run inference
        '''
        if len(inputs) < len(self.sess.get_inputs()):
            raise ValueError(f'{inputs} does not match model requirement')

        inp_dict = {}
        for i, inp in enumerate(self.sess.get_inputs()):
            inputs[i] = inputs[i].astype(type_map[inp.type])
            if inp.shape != list(inputs[i].shape):
                try:
                    inputs[i] = inputs[i].reshape(inp.shape)
                except:
                    raise ValueError(f'Shape mismatch {inp.shape} versus {inputs[i].shape}')
            inp_dict[inp.name] = inputs[i]

        output_names = []
        for i in range(len(self.sess.get_outputs())):
            output_names.append(self.sess.get_outputs()[i].name)

        res = self.sess.run(output_names, inp_dict)
        return res


class RknnModelContainer():
    ''' A wrapper that initialize RKNN inference environment
    '''
    def __init__(self, model_path, target=None, device_id=None) -> None:

        rknn = RKNNLite()
        rknn.load_rknn(model_path)
        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        self.rknn = rknn 

    def run(self, inputs):
        ''' Run inference on NPU chip
        '''
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            inputs = [inputs]
        result = self.rknn.inference(inputs=inputs)
        return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model_path', type=str, required= True, 
                        help='model path, could be .onnx or .rknn file')
    parser.add_argument('-t', '--target', type=str, default=None, 
                        help='target RKNPU platform')
    parser.add_argument('-d', '--device_id', type=str, default=None, 
                        help='device id')
    parser.add_argument('-m', '--mode', type=str, choices=['npu', 'onnx'], nargs='?', default='npu',
                        help='inference mode with KRNPU or ONNX on CPU')
    parser.add_argument('-i', '--img_path', type=str, default=None, required=True, 
                        help='Path to image file')
    parser.add_argument('-o', '--out_path', type=str, default='output.jpg', 
                        help='Path to write result image')
    parser.add_argument('-a', '--anchors', type=str, default='./models/anchors_yolov5.txt', 
                        help='target to anchor file, only yolov5, yolov7 need this param')

    args = parser.parse_args()

    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3, -1, 2).tolist()
        print("use anchors from '{}', which is {}".format(args.anchors, anchors))
    
    if args.mode == 'npu':
        model = RknnModelContainer(
                    args.model_path, 
                    args.target, 
                    args.device_id)
    elif args.mode == 'onnx':
        model = OnnxModelContainer(args.model_path)

    img_path = os.path.join(args.img_path)
    img_src = cv2.imread(img_path)

    co_helper = COCO_test_helper(enable_letter_box=True)
    # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
    pad_color = (0, 0, 0)
    img = co_helper.letter_box(
            im=img_src.copy(), 
            new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args.mode == 'onnx':
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape).astype(np.float32)
        img = img / 255.

    input_data = img[np.newaxis, ...]
    outputs = model.run(input_data)
    boxes, classes, scores = post_process(outputs, anchors)

    if boxes is not None:
        draw(img_src, co_helper.get_real_box(boxes), scores, classes)
    cv2.imwrite(args.out_path, img_src)
