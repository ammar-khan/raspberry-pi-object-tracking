##
# Copyright 2018, Ammar Ali Khan
# Licensed under MIT.
##

import time
import cv2
import numpy as np
from src.common.package.config import application as config_app
from src.opencv.package.config import application as config_opencv
from src.common.package.http import server as http_server
from src.common.package.http.handler import Handler
from src.common.package.camera.handler import Handler as camera
from src.common.package.frame.handler import Handler as frame_handler
from src.common.package.tracker.handler import Handler as Tracker
from src.opencv.package.opencv.handler import Handler as OpenCV

# Constant
opencv = OpenCV()
tracker = Tracker()


##
# StreamHandler class - inherit Handler
# This class provide handler for HTTP streaming
# Note: this class should override Handler.stream
##
class StreamHandler(Handler):

    ##
    # Override method Handler.stream()
    ##
    def stream(self):
        Handler.stream(self)
        print('[INFO] Overriding stream method...')

        # Initialise capture
        capture = camera(src=config_app.CAPTURING_DEVICE,
                         use_pi_camera=config_app.USE_PI_CAMERA,
                         resolution=config_app.RESOLUTION,
                         frame_rate=config_app.FRAME_RATE)

        if config_app.USE_PI_CAMERA:
            print('[INFO] Warming up pi camera...')
        else:
            print('[INFO] Warming up camera...')

        time.sleep(2.0)

        print('[INFO] Start capturing...')
        while True:
            # Read a frame from capture
            frame = capture.read()

            # Get frame dimensions
            (height, width) = frame.shape[:2]

            # OpenCV face detections
            detections = opencv.dnn_face_detector(frame=frame, scale_factor=1.0)

            # Reset trackers
            tracker.reset_trackers()

            # If returns any detection
            for i in range(0, detections.shape[2]):
                # Get confidence associated with the detection
                confidence = detections[0, 0, i, 2]

                # Filter weak detection
                if confidence < config_opencv.CONFIDENCE:
                    continue

                # Calculate coordinates
                box = detections[0, 0, i, 3:7] * np.array([width,
                                                           height,
                                                           width,
                                                           height])

                # List of bounding box rectangles
                (left, top, right, bottom) = box.astype('int')

                # Set coordinates
                coordinates = {'left': left,
                               'top': top,
                               'right': right,
                               'bottom': bottom}

                # Create or update tracker
                tracker_id = tracker.track(coordinates)

                # Create box with id or description
                frame = frame_handler.rectangle(frame=frame,
                                                coordinates=coordinates,
                                                text='Person {}'.format(tracker_id))

            # Remove unused trackers
            tracker.remove_trackers()

            # Write date time on the frame
            frame = frame_handler.text(frame=frame,
                                       coordinates={'left': config_app.WIDTH - 150, 'top': config_app.HEIGHT - 20},
                                       text=time.strftime('%d/%m/%Y %H:%M:%S', time.localtime()),
                                       font_color=(0, 0, 255))

            # Convert frame into buffer for streaming
            retval, buffer = cv2.imencode('.jpg', frame)

            # Write buffer to HTML Handler
            self.wfile.write(b'--FRAME\r\n')
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', len(buffer))
            self.end_headers()
            self.wfile.write(buffer)
            self.wfile.write(b'\r\n')


##
# Method main()
##
def main():
    try:
        # Prepare and start HTTP server
        address = ('', config_app.HTTP_PORT)
        server = http_server.Server(address, StreamHandler)
        print('[INFO] HTTP server started successfully at %s' % str(server.server_address))
        print('[INFO] Waiting for client to connect to port %s' % str(config_app.HTTP_PORT))
        server.serve_forever()
    except Exception as e:
        server.socket.close()
        print('[INFO] HTTP server closed successfully.')
        print('[ERROR] Exception: %s' % str(e))


if __name__ == '__main__':
    main()
