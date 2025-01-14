import pyrealsense2 as rs
import numpy as np

# Filter initialisieren und Optionen setzen
# Filter initialisieren und Optionen setzen
decimation_filter = rs.decimation_filter()
decimation_filter.set_option(rs.option.filter_magnitude, 2)
# Threshold Filter
threshold_filter = rs.threshold_filter()
threshold_filter.set_option(rs.option.min_distance, 0.1)
threshold_filter.set_option(rs.option.max_distance, 4.0)
# HDR Merge (wird aktiviert, wenn HDR-Merge-fähige Frames verwendet werden)
hdr_merge = rs.hdr_merge()
# Depth to Disparity
depth_to_disparity = rs.disparity_transform(True)  # True = Depth to Disparity
# Spatial Filter
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
spatial_filter.set_option(rs.option.holes_fill, 1)  # Hole Filling Mode: Disabled
# Temporal Filter
temporal_filter = rs.temporal_filter()
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
temporal_filter.set_option(rs.option.holes_fill, 2)  # Persistency Mode: Valid in 2/last 2
# Hole Filling Filter
hole_filling_filter = rs.hole_filling_filter()
hole_filling_filter.set_option(rs.option.holes_fill, 2)  # Hole Filling Mode: Farthest from all neighbors
# Disparity to Depth
disparity_to_depth = rs.disparity_transform(False)  # False = Disparity to Depth


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

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Tiefenbild
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB-Bild

        # Start streaming
        self.pipeline.start(config)

        # Kameraparameter auslesen
        self.profile = self.pipeline.get_active_profile()
        self.rgb_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()

        self.rgb_intrinsics = self.rgb_stream.get_intrinsics()
        self.depth_intrinsics = self.depth_stream.get_intrinsics()
        self.extrinsics = self.depth_stream.get_extrinsics_to(self.rgb_stream)

        # PointCloud-Objekt
        self.pc = rs.pointcloud()
        self.points = rs.points()

        # Transformation Matrix aus Extrinsik (Tiefenkamera -> RGB-Kamera)
        self.rotation_matrix = np.array(self.extrinsics.rotation).reshape(3, 3)
        self.translation_vector = np.array(self.extrinsics.translation)

        # Verzerrungskoeffizienten für RGB-Kamera auslesen
        self.rgb_dist_coeffs = np.array(self.rgb_intrinsics.coeffs).reshape(5, 1)

        # Kameramatrix für RGB-Kamera
        self.rgb_camera_matrix = np.array([
            [self.rgb_intrinsics.fx, 0, self.rgb_intrinsics.ppx],
            [0, self.rgb_intrinsics.fy, self.rgb_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=float)


    def get_frame(self):
        frames = self.pipeline.wait_for_frames()  # Frames abrufen
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Filter anwenden
        # depth_frame = decimation_filter.process(depth_frame)
        depth_frame = hdr_merge.process(depth_frame)
        depth_frame = threshold_filter.process(depth_frame)
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)

        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, depth_image, color_image

    def pixel_to_world(self, pixel, depth):
        """
        Transformiert Pixel- und Tiefenwerte (RGB-Bild) in Weltkoordinaten.
        """
        u, v = pixel
        z = depth

        # Von Pixel- zu Kamerakoordinaten (RGB-Intrinsik verwenden)
        x_camera = (u - self.rgb_intrinsics.ppx) / self.rgb_intrinsics.fx * z
        y_camera = (v - self.rgb_intrinsics.ppy) / self.rgb_intrinsics.fy * z
        camera_point = np.array([x_camera, y_camera, z])

        # Transformation in den Tiefenkamera-Frame
        world_point = self.rotation_matrix @ camera_point + self.translation_vector
        return world_point

    def get_pointcloud(self):
        # Hole die aktuellen Frames
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Berechne die Punktwolke
        self.pc.map_to(color_frame)  # Mapping der Farbe auf die Punktwolke
        self.points = self.pc.calculate(depth_frame)

        # Extrahiere die Punktwolken-Daten (XYZ und RGB)
        vertices = np.asanyarray(self.points.get_vertices())  # XYZ-Koordinaten
        tex_coords = np.asanyarray(self.points.get_texture_coordinates())  # Texturkoordinaten (für Farben)

        return vertices, tex_coords

    def release(self):
        self.pipeline.stop()  # Pipeline stoppen
