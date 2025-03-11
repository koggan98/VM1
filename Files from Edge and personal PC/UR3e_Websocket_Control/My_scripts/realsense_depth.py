# habe ich von Pysource runtergeladen 
import pyrealsense2 as rs
import numpy as np

# Filter initialisieren
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

temporal_filter = rs.temporal_filter()
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.1)
temporal_filter.set_option(rs.option.filter_smooth_delta, 100)

hole_filling_filter = rs.hole_filling_filter()

threshold_filter = rs.threshold_filter()
threshold_filter.set_option(rs.option.min_distance, 0.3)
threshold_filter.set_option(rs.option.max_distance, 3.0)


class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Captures depth frames at 30 frames per second in 16bit format
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Captures RGB color frames at 30 frames per second in BGR 8bit format

        # Start streaming
        self.pipeline.start(config)


    def get_frame(self):
        frames = self.pipeline.wait_for_frames() # Retrieve latest frame from camera pipeline
        depth_frame = frames.get_depth_frame() # Extracts individial depth frames from frames
        color_frame = frames.get_color_frame() # Extracts individial color frames from frames

        # Wende die Filter an
        depth_frame = threshold_filter.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)
           
        depth_image = np.asanyarray(depth_frame.get_data()) # Converts frames in np arrays
        color_image = np.asanyarray(color_frame.get_data()) # Converts frames in np arrays
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image # True if frames succesfully captured, depth_image = 2D array of depth values, color_image = 3 Channel color image


    def release(self):
        self.pipeline.stop() # Stops pipeline
