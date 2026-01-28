import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
logger = logging.getLogger(__name__)


# calibrator
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)       
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        batch = self.stream.next_batch()
        if not batch.size:   
            return None
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)
