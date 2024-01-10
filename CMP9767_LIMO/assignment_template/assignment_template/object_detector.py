import rclpy
from rclpy.node import Node
from rclpy import qos
import numpy as np

# OpenCV
import cv2
from cv2 import inRange
# ROS libraries
import image_geometry
from tf2_ros import Buffer, TransformListener

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_geometry_msgs import do_transform_pose
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray

class ObjectDetector(Node):
    camera_model = None
    image_depth_ros = None

    visualisation = True

    # aspect ration between color and depth cameras
    # calculated as (color_horizontal_FOV/color_width) / (depth_horizontal_FOV/depth_width) from the dabai camera parameters

    #color2depth_aspect = (71.0/640) / (67.9/400) # for real robot
    color2depth_aspect = 1 #for simulation

    def __init__(self):    
        super().__init__('pothole_finder')
        self.bridge = CvBridge()

        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)

        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                  self.image_color_callback, qos_profile=qos.qos_profile_sensor_data)
        
        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', 
                                                  self.image_depth_callback, qos_profile=qos.qos_profile_sensor_data)
        
        self._pothole_publisher_ = self.create_publisher(MarkerArray, '/limo/Pothole_location', 10)



        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=20))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.centroid_coordinates = []
        self.pothole_points = []
        self.clusters = 0
    

    def get_tf_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None


    def camera_info_callback(self, data):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(data)

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def image_color_callback(self, data):

        # wait for camera_model and depth image to arrive
        if self.camera_model is None:
            return

        if self.image_depth_ros is None:
            return

        # covert images to open_cv
        try:
            image_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        except CvBridgeError as e:
            print(e)


        # detect a color blob in the color image
        hsv = cv2.cvtColor(image_color,cv2.COLOR_BGR2HSV)
        mask = inRange(hsv, (140, 82, 0), (179, 255, 255))                                            # this is the range for pink potholes
        kernel = np.ones((5, 5), np.uint8)                                                            # use this kernel for noise reduction
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)                                         # open
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)                                        # close 
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)              # find contours (essentially blobs)
        
        for j in contours:                                                                            # for each blob
            # Calculate centroid
            M = cv2.moments(j)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])                                                         # here we find x 
                cY = int(M["m01"] / M["m00"])                                                         # and y of centroid 
            else:
                cX, cY = 0, 0
                print('no object in frame')
    
            self.centroid_coordinates.append((cX,cY))

           # Draw bounding box and centroid
            #cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.circle(image_color, (cX, cY), 5, (0, 0, 255), -1)

      
        for centroid in self.centroid_coordinates:
            centreX, centreY = centroid

            # calculate the y,x centroid
            image_coords = (centreY), (centreX)        
        
        # "map" from color to depth image
            depth_coords = (image_depth.shape[0]/2 + (image_coords[0] - image_color.shape[0]/2)*self.color2depth_aspect, 
                image_depth.shape[1]/2 + (image_coords[1] - image_color.shape[1]/2)*self.color2depth_aspect)
        
        # get the depth reading at the centroid location
            depth_value = image_depth[int(depth_coords[0]), int(depth_coords[1])]                     

            if 0 < depth_value < 1.2:                                                                     # filter out depth anomolies

        # calculate object's 3d location in camera coords 
                camera_coords = self.camera_model.projectPixelTo3dRay((image_coords[1], image_coords[0])) # project the image coords (x,y) into 3D ray in camera coords 
                camera_coords = [x/camera_coords[2] for x in camera_coords]                               # adjust the resulting vector so that z = 1
                camera_coords = [x*depth_value for x in camera_coords]                                    # multiply the vector by depth


            
        #define a point in camera coordinates
            
                object_location = PoseStamped()                                                           # take the pose of detected pothole for processing
                object_location.header.frame_id = "depth_link"
                object_location.pose.orientation.w = 1.0
                object_location.pose.position.x = camera_coords[0]
                object_location.pose.position.y = camera_coords[1]
                object_location.pose.position.z = camera_coords[2]

            # Marker array for visualisation 
            Markers = MarkerArray()
            marker_id = 0
            Markers.markers = []
            # Transform to the 'map' frame
            transform_map = self.get_tf_transform('odom', 'depth_link')                                # Transforming to the 'map' frame
            if transform_map:
                p_map = do_transform_pose(object_location.pose, transform_map)
                self.pothole_points.append((p_map.position.x, p_map.position.y))                      # take the map points x & y
            
               # Extracting coordinates from PointStamped messages
                
                points = np.array(self.pothole_points)                                                # create a numpy array for clustering
                
                # DBSCAN clustering       
                dbscan = DBSCAN(eps=0.132, min_samples=2).fit(points)                                 # I use DBSCAN to cluster, this removes..
                labels = dbscan.labels_                                                               # ..duplicates and duplicates caused by drift

                #print('x and y of potholes = ',labels)                                               # print if you want points

                self.clusters = len(set(labels)) - (1 if -1 in labels else 0)                         # find the length of the list (how many potholes)
               
                # Getting unique labels (excluding noise label -1)
                unique_labels = set(labels) - {-1}                                                    # details of each element minus duplicates



                                                                                  # clear list each run
                


                
                # Calculating centroids for each cluster
                for label in unique_labels:

                # Filter points belonging to the current cluster
                    cluster_points = points[labels == label]
                    centroid_db = np.mean(cluster_points, axis=0)                                     # get the centroid of each cluster
                    
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.type = Marker.SPHERE                                                       # Marker shape in rviz
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.2                                                              # Marker size
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2
                    marker.color.a = 1.0                                                              # Alpha channel (transparency)
                    marker.color.r = 1.0                                                              # Red color
                    marker.color.g = 0.0
                    marker.color.b = 0.0


                    #fill my marker array with estimated positions
                    marker.id = marker_id
                    marker.pose.position.x = centroid_db[0]
                    marker.pose.position.y = centroid_db[1]
                    marker.pose.position.z = 0.0                                                      # Assuming z = 0 for visualization
                    Markers.markers.append(marker)
                    marker_id += 1
                    self._pothole_publisher_.publish(Markers)
            
            else:
                self.get_logger().warning("Transform between 'depth_link' and 'map' not available.")
        print('the number of potholes in this list is: = ', self.clusters)
 
        
        #resize and adjust for visualisation
        #image_color = cv2.resize(image_color, (0,0), fx=0.8, fy=0.8)                                     # Maybe i want to show camera image for comparison
        #image_depth *= 1.0/10.0 # scale for visualisation (max range 10.0 m)
        
        #cv2.imshow("image color", image_color)
        #cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)
    pothole_finder = ObjectDetector()
    rclpy.spin(pothole_finder)
    pothole_finder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
