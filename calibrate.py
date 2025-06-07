import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import cv2

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_list, input_shape=(3, 224, 224), batch_size=8, cache_file="calib_cache.bin"):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file

        with open(calib_list, 'r') as f:
            self.image_paths = [line.strip() for line in f if line.strip()]
        self.current_index = 0

        nbytes = int(batch_size * np.prod(input_shape) * np.float32().nbytes)
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.current_index:self.current_index + self.batch_size]
        batch_data = []

        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC → CHW
            batch_data.append(img)

        actual_batch_size = len(batch_data)
        batch_np = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        batch_np[:actual_batch_size] = np.stack(batch_data)

        cuda.memcpy_htod(self.device_input, batch_np)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

def build_int8_engine(onnx_path, calibrator, max_workspace_size=1<<30):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    # config.set_flag(trt.BuilderFlag.INT8)
    # config.int8_calibrator = calibrator  # Warning: Deprecated in TRT 10.1+

    # print("Building INT8 engine...")
    # serialized_engine = builder.build_serialized_network(network, config)
        config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator  # Deprecated, but still works in 10.1

    # Add optimization profile for dynamic input
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name,
                      min=(1, 3, 224, 224),
                      opt=(8, 3, 224, 224),
                      max=(16, 3, 224, 224))
    config.add_optimization_profile(profile)

    print("Building INT8 engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build serialized engine.")
        return None

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("Engine built successfully!")
    return engine

if __name__ == "__main__":
    onnx_model_path = "model.onnx"
    calib_list_path = "calib_list.txt"
    cache_file = "calib_cache.bin"
    engine_file = "model_int8.engine"

    input_shape = (3, 224, 224)  # CHW
    batch_size = 8

    calibrator = MyCalibrator(calib_list_path, input_shape, batch_size, cache_file)
    engine = build_int8_engine(onnx_model_path, calibrator)

    if engine:
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())
        print(f"INT8 engine saved to {engine_file}")
