import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import cv2
import time
from datetime import datetime
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
    
    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment the frame count
    user_data.increment()

    # Extract video frame details (format, width, height)
    format, width, height = get_caps_from_pad(pad)

    # Get detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Loop through detections to find recognized faces
    for detection in detections:
        print(detection)
        #if True:  # Example method for checking face detection
            # Access metadata for the recognized person via HailoGallery
           # recognized_person = detection.get_metadata('recognized_person')  # Replace with the correct method from API
           # if recognized_person:
           #     person_name = recognized_person.get_name()  # Assuming the API has a method to fetch the person's name
           #     print(f"Recognized Person: {person_name}")

    return Gst.PadProbeReturn.OK
    

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3 
        nms_iou_threshold = 0.45

        self.model_dir = "/home/pi/work/ai/models"
        self.tappas_worspace = "/home/pi/work/ai/tappas/tappas"
        self.postprocess_dir = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes"

        #detection
        self.face_detection_hef =  self.model_dir + '/retinaface_mobilenet_v1.hef'
        self.face_detection_so = self.postprocess_dir + '/libface_detection_post.so'
        self.face_detection_function_name = "retinaface"

        #recognition
        self.recognition_hef =  self.model_dir + '/arcface_mobilefacenet.hef'
        self.recognition_so = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libface_recognition_post.so"
        self.recognition_function_name =  "arcface_nv12"

        self.local_gallery_file = "/home/pi/work/ai/exp/face_recognition_local_gallery_rgba.json"
        self.cropper_so = self.postprocess_dir + '/cropping_algorithms/libvms_croppers.so'
        self.vdevice_key = 1

        #align
        self.facealign_so = "/usr/lib/aarch64-linux-gnu/apps/vms/libvms_face_align.so"

        self.video_sink_element = "xvimagesink"
        
        # Temporary code: new postprocess will be merged to TAPPAS.
        # Check if new postprocess so file exists
        new_postprocess_path = os.path.join(self.current_path, '../resources/libyolo_hailortpp_post.so')
        if os.path.exists(new_postprocess_path):
            self.default_postprocess_so = new_postprocess_path
        else:
            self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolo_hailortpp_post.so')

        if args.hef_path is not None:
            self.hef_path = args.hef_path
        # Set the HEF file path based on the network
        elif args.network == "yolov6n":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')
        elif args.network == "yolov8s":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
        elif args.network == "yolox_s_leaky":
            self.hef_path = os.path.join(self.current_path, '../resources/yolox_s_leaky_h8l_mz.hef')
        else:
            assert False, "Invalid network type"

        # User-defined label JSON file
        if args.labels_json is not None:
            self.labels_config = f' config-path={args.labels_json} '
            # Temporary code
            if not os.path.exists(new_postprocess_path):
                print("New postprocess so file is missing. It is required to support custom labels. Check documentation for more information.")
                exit(1)
        else:
            self.labels_config = ''

        self.app_callback = app_callback
    
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        #setproctitle.setproctitle("Hailo Detection App")

        self.create_pipeline()

    def get_pipeline_string(self):
        if self.source_type == "rpi":
            source_element = (
                "libcamerasrc name=src_0 auto-focus-mode=0 ! "
                f"video/x-raw,format=YUY2,width=800,height=600,framerate=30/1 ! "
                + QUEUE("queue_pre_flip",max_size_buffers=30)
                + "videoflip video-direction=horiz ! "
            )
        elif self.source_type == "usb":
            source_element = (
                f"v4l2src device={self.video_source} name=src_0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
            )
        else:
            source_element = (
                f"filesrc location={self.video_source} name=src_0 ! "
                + QUEUE("queue_dec264")
                + " qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
                " video/x-raw, format=I420 ! "
            )
        #source_element += QUEUE("hailo_pre_convert_q", max_size_buffers=20, leaky="no")
        #source_element += "videoscale n-threads=2 qos=false ! "

        face_detection_pipeline = (
            f"hailonet hef-path={self.face_detection_hef} scheduling-algorithm=1 vdevice-key={self.vdevice_key} ! "
            + QUEUE("detector_post_q",max_size_buffers=30,leaky="no")
            + f"hailofilter so-path={self.face_detection_so} name=face_detection_hailofilter qos=false function_name={self.face_detection_function_name} ! "
        )

        detector_pipeline = (
            "tee name=t hailomuxer name=hmux "
            +"t. ! "
            + QUEUE("detector_bypass_q", max_size_buffers=30, leaky="no")
            + "hmux. t. ! "
            + "videoscale name=face_videoscale method=0 n-threads=2 add-borders=false qos=false ! "
            + "video/x-raw, pixel-aspect-ratio=1/1 ! "
            + QUEUE("pre_face_detector_infer_q ", max_size_buffers=30, leaky="no")
            + face_detection_pipeline
            + QUEUE("post_filter", max_size_buffers=30, leaky="no")
            + "hmux. hmux. ! "
        )

        face_tracker_pipeline = (
            "hailotracker name=hailo_face_tracker class-id=-1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 "
           +"keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 keep-past-metadata=true qos=false ! "
        )

        cropper_pipeline = (
            f"hailocropper so-path={self.cropper_so} function-name=face_recognition internal-offset=true name=cropper2 "
           +"hailoaggregator name=agg2 cropper2. ! "
           +QUEUE("bypess2_q", max_size_buffers=30, leaky="no")
           +"agg2. cropper2. ! "
        )

        align_pipeline = (
            f"hailofilter so-path={self.facealign_so} name=face_align_hailofilter use-gst-buffer=true qos=false ! "
        )

        recognition_pipeline = (
            f"hailonet hef-path={self.recognition_hef} scheduling-algorithm=1 vdevice-key={self.vdevice_key} ! "
           +QUEUE("recognition_pre_agg_q", max_size_buffers=30, leaky="no")
           +f"hailofilter function-name={self.recognition_function_name} so-path={self.recognition_so} name=face_recognition_hailofilter qos=false ! "
           +QUEUE("recognition_post_agg_q", max_size_buffers=30, leaky="no")
           +"agg2. agg2. ! "
        )

        gallery_pipeline = (
            f"hailogallery gallery-file-path={self.local_gallery_file} "
            +"load-local-gallery=true similarity-thr=.4 gallery-queue-size=20 class-id=-1 ! "
        )
        

#config-path=/home/pi/work/ai/tappas/tappas/apps/h8/gstreamer/general/face_recognition/resources/configs/scrfd.json
        pipeline_string = (
            #"hailomuxer name=hmux "
            source_element
            + QUEUE("hailo_pre_convert_q", max_size_buffers=30, leaky="no")   
            + "videoconvert n-threads=2 qos=false ! "
            #+ "tee name=t ! "
            + QUEUE("pre_detector_q", max_size_buffers=30, leaky="no")           
            + detector_pipeline
            + QUEUE("pre_tracker_q", max_size_buffers=30, leaky="no")
            + face_tracker_pipeline
            + QUEUE("pre_copper_q", max_size_buffers=30, leaky="no")
            + cropper_pipeline
            + QUEUE("pre_align_q", max_size_buffers=30, leaky="no")
            + align_pipeline
            + QUEUE("pre_recognition_pipeline", max_size_buffers=30, leaky="no")
            + recognition_pipeline
            + QUEUE("hailo_pre_gallery_q", max_size_buffers=30, leaky="no")
            + gallery_pipeline
            + QUEUE("hailo_pre_draw_q", max_size_buffers=30, leaky="no")
            + "hailooverlay name=hailo_overlay qos=false show-confidence=true local-gallery=true line-thickness=5 font-thickness=2 landmark-point-radius=8 ! "
            + QUEUE("queue_user_callback")
            + "identity name=identity_callback ! "

            + QUEUE("hailo_post_draw", max_size_buffers=30, leaky="no")
            + "videoconvert n-threads=4 qos=false name=display_videoconvert qos=false ! "
            + QUEUE("hailo_display_q", max_size_buffers=30, leaky="no")
            + f"fpsdisplaysink video-sink={self.video_sink_element} name=hailo_display sync=false text-overlay=false "
        )
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    parser = get_default_parser()
    # Add additional arguments here
    parser.add_argument(
        "--network",
        default="yolov6n",
        choices=['yolov6n', 'yolov8s', 'yolox_s_leaky'],
        help="Which Network to use, default is yolov6n",
    )
    parser.add_argument(
        "--hef-path",
        default=None,
        help="Path to HEF file",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Path to costume labels JSON file",
    )
    args = parser.parse_args()
    app = GStreamerDetectionApp(args, user_data)
    app.run()
