import glob, os, cv2
import numpy as np
import tensorrt as trt
from calibrator import Calibrator


height = 640
width = 640
image_path = 'images'
model_path = "build/yolov5n_fp32.onnx"
engine_model_path = "build/yolov5n_int8.engine"
calibration_table = 'build/yolov5n_int8_calibration.cache'

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) 


def preprocess(image):
    h, w, c = image.shape
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    image = cv2.resize(image, (tw, th))
    image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114))  
    image = image / 255.0
    image = image[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)
    return image


class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = 12
        self.batch_size = 8
        self.img_list = glob.glob(os.path.join(image_path, "*.jpg"))
        assert len(self.img_list) >= self.batch_size * self.length
        self.calibration_data = np.zeros((self.batch_size, 3, height, width), dtype=np.float32)

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data[i] = img
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


def get_engine(onnx_file_path="", engine_file_path="", calibration_stream=None, calibration_table_path=""):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:      
        if not os.path.exists(onnx_file_path):
            quit('ONNX file {} not found'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())
            assert network.num_layers > 0, 'Failed to parse ONNX model. Please check if the ONNX model is compatible '      
        #builder.max_batch_size = 1
        config = builder.create_builder_config()
        #config.max_workspace_size = 1 << 32
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
        config.set_flag(trt.BuilderFlag.INT8)
        assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
        config.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
        runtime = trt.Runtime(TRT_LOGGER)
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            print('Failed to create the engine')
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    get_engine(model_path, engine_model_path, calibration_stream=DataLoader(), calibration_table_path=calibration_table)
    
    
