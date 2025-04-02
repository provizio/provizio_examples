from threading import Event, Condition
import time
import datetime
import sys
import signal
import provizio_dds
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class dotdict(dict):  # noqa
    """
    dot.notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ExampleDDS:
    """
    Receives pointcloud, odometry and entities through DDS.
    Publishes a filtered list of entities, as an example.
    """

    INPUT_PCL_TOPIC = 'rt/provizio_radar_point_cloud'
    INPUT_ODOMETRY_TOPIC = 'rt/provizio_radar_odometry'
    INPUT_ENTITIES_TOPIC = 'rt/provizio_entities'
    OUTPUT_ENTITIES_TOPIC = 'rt/provizio_entities_filtered'

    FPS = 10

    YAW_RATE_JUMP_ALLOWED = (
        0.5  # Maximum allowed jumping between consective calculated yaw rate
    )
    FAIL_ON_NO_INPUT_FOR_SECONDS = 30
    MISS_THRESHOLD = 5

    def __init__(self) -> None:

        #  DDS
        self.stop_request = Event()
        self.process_without_subscribers = True
        self.has_subscribers = False
        self.last_frame = None
        self.last_index = 0
        self.cv = Condition() # Used for thread-syncing with background threads of provizio_dds

        self._last_input_at = datetime.datetime.now()

        # Odometry Service
        self._last_odometry_input_at = datetime.datetime.now()
        self.prev_yaw_angle = None

        # Odometry function related params and buffers
        self._last_odometry_frame = None
        self._last_odometry_index = 0
        self.prev_last_odometry_index = 0
        self.cnt_misses = 0
        self.fps = ExampleDDS.FPS
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.ego_position = np.array([0, 0])

        # DDS
        dds_domain_participant = provizio_dds.make_domain_participant() 
        
        # Subscriber for Pointcloud from the radar
        # We use the standard ROS sensor_msgs/msg/PointCloud2 with Radar specific Fields
        # See https://github.com/provizio/provizio_dds_idls/blob/master/TOPICS.md
        self._dds_pcl_subscriber = provizio_dds.Subscriber(
            dds_domain_participant,
            ExampleDDS.INPUT_PCL_TOPIC,
            provizio_dds.PointCloud2PubSubType,
            provizio_dds.PointCloud2,
            lambda data: self.on_pointcloud(data),
        )
        
        # Subscriber for Odometry
        # We use the standard ROS nav_msgs/msg/Odometry
        # See https://github.com/provizio/provizio_dds_idls/blob/master/TOPICS.md
        self._dds_odometry_subscriber = provizio_dds.Subscriber(
            dds_domain_participant,
            ExampleDDS.INPUT_ODOMETRY_TOPIC,
            provizio_dds.OdometryPubSubType,  # DDS Pub/Sub Type
            provizio_dds.Odometry,
            lambda data: self.on_odometry_msg(data),
        )
        
        # Subscriber for Entities (tracked objects)
        # We use the standard ROS sensor_msgs/msg/PointCloud2 as a transport type for Entities
        # See https://github.com/provizio/provizio_dds_idls/blob/master/TOPICS.md
        self._dds_entities_subscriber = provizio_dds.Subscriber(
            dds_domain_participant,
            ExampleDDS.INPUT_ENTITIES_TOPIC,
            provizio_dds.PointCloud2PubSubType,
            provizio_dds.PointCloud2,
            lambda data: self.on_entities_msg(data),
        )

        # Publisher for Entities (tracked objects)
        # We use Pointcloud2 as a transport type for Entities
        # See https://github.com/provizio/provizio_dds_idls/blob/master/TOPICS.md
        self._dds_entities_publisher = provizio_dds.Publisher(
            dds_domain_participant,
            ExampleDDS.OUTPUT_ENTITIES_TOPIC,
            provizio_dds.PointCloud2PubSubType,
            lambda _, has_subscribers: self.on_has_subscribers(has_subscribers),
        )

        logger.info("Waiting for Pointcloud...")


    def on_odometry_msg(self, odometry_msgs: provizio_dds.Odometry) -> None:
        """
        Callback triggered when an odometry message is received.
        """
        self._last_odometry_index += 1
        logger.debug(f"Received a new Odometry MSG")

        with self.cv:
            # Extract yaw rate and ego vehicle velocity from the odometry message
            twist = odometry_msgs.twist().twist()
            yaw_rate = twist.angular().z()
            ego_velocity = np.linalg.norm([twist.linear().x(), twist.linear().y()])

            # Compute delta position based on velocity and yaw rate
            dt = 1 / self.FPS
            linear_velocity_x_y = ego_velocity * np.array([np.cos(yaw_rate * dt), np.sin(yaw_rate * dt)])
            delta_position = linear_velocity_x_y * dt

            self.cv.notify_all()
            logger.debug(f"Current ego velocity: {ego_velocity} | Current ego yaw_rate: {yaw_rate} | Delta position: {delta_position}")

    
    def on_pointcloud(self, pcl: provizio_dds.PointCloud2):
        """
        Callback triggered when a new point cloud message is received.
        """
        cloud = provizio_dds.point_cloud2.read_points_numpy(pcl)
        self.last_index += 1

        # Store received point cloud in a structured format
        frame = dotdict(
            index=self.last_index,
            timestamp_s=pcl.header().stamp().sec(),
            timestamp_ns=pcl.header().stamp().nanosec(),
            frame_id=pcl.header().frame_id(),
            num_points=len(cloud),
            cloud=cloud,
        )

        with self.cv:
            self._last_input_at = datetime.datetime.now()
            self.last_frame = frame
            self.cv.notify_all()

        logger.debug(f"Received pointcloud {self.last_index}")

    def on_entities_msg(self, dds_entities_msg) -> None:
        """
        Callback triggered when an entities message (tracked objects) is received.
        """
        with self.cv:

            # copy header info
            radar_id = dds_entities_msg.header().frame_id()

            # read entities as list
            entities = provizio_dds.point_cloud2.read_points_list(dds_entities_msg)
            logger.debug(f"Received {len(entities)} entities on {radar_id}")

            for entity in entities:
                # logger some info about entities received.
                # All the entities fields are described there https://github.com/provizio/provizio_dds_idls/blob/master/TOPICS.md
                logger.debug(f"Detected {entity.entity_class} in {'%.2f' % entity.x}, {'%.2f' % entity.y} with ground velocity {'%.2f' % entity.ground_relative_radial_velocity}")

            PEDESTRIAN_ENTITY_CLASS = 1
            # Filter the entities to only send back pedestrians
            filtered_entities = filter(lambda x: x.entity_class == PEDESTRIAN_ENTITY_CLASS, entities)

            self.send_filtered_entities(filtered_entities)
               
            self.last_radar_entities_at = datetime.datetime.now()
            self.cv.notify_all()
            

    def on_has_subscribers(self, has_subscribers: bool):
        """
        Triggered when detecting a new DDS subscriber
        """
        with self.cv:
            self.has_subscribers = has_subscribers
            self.cv.notify_all()
            logger.info("New subscriber detected")


    def send_filtered_entities(self, entities: list):
        """
        Example of how to publish entities using provizio_DDS. 
        """
        dds_entities_data = provizio_dds.point_cloud2.make_radar_entities(
            provizio_dds.point_cloud2.make_header(
                self.last_frame.timestamp_s, self.last_frame.timestamp_ns, self.last_frame.frame_id
            ),
            entities,
        )
        success = self._dds_entities_publisher.publish(dds_entities_data)
        if success:
            logger.debug(f"Filtered Entities for frame {self.last_frame.index} published successfully")

            
    def run(self) -> None:
        """
        Main Loop, waits for new pointcloud to be received and then runs the process.
        """
        single_wait_timeout = (
            1  # We'll be check for stop_request every single_wait_timeout seconds
        )

        with self.cv:
            current_frame_index = (
                -1 if self.last_frame is None else self.last_frame.index
            )

        while not self.stop_request.is_set():
            frame = None
            with self.cv:
                if self.has_subscribers or self.process_without_subscribers:
                    if self.cv.wait_for(
                        lambda: (
                            self.has_subscribers or self.process_without_subscribers
                        )
                        and self.last_frame is not None
                        and self.last_frame.index != current_frame_index,
                        single_wait_timeout,
                    ):
                        # Got a new frame and at least one subscriber
                        frame = self.last_frame
                        current_frame_index = frame.index

                else:
                    # No subscribers yet, i.e. no need to process received point clouds
                    self.cv.wait_for(lambda: self.has_subscribers, single_wait_timeout)

            if frame is not None:
                self.process_frame(frame)
            elif (
                datetime.datetime.now() - self._last_input_at
            ).total_seconds() > self.FAIL_ON_NO_INPUT_FOR_SECONDS:
                print(
                    f"Exiting: no input data for over {self.FAIL_ON_NO_INPUT_FOR_SECONDS} sec"
                )
                sys.exit(1)

        self.cleanup()


    def process_frame(self, frame: dotdict):
        """
        Processing pointcloud placeholder. 
        Simple example of accessing the pointcloud transformed previously into a numpy array.
        """
        start = time.perf_counter()
        
        # Here we consumme the pointcloud received
        logger.debug(f"Received {len(frame.cloud)} points. First point is in {'%.2f' % frame.cloud[0][0]}, {'%.2f' % frame.cloud[0][1]}, {'%.2f' % frame.cloud[0][2]}")
        
        logger.debug(
            f"Frame #{frame.index} Process time: {(time.perf_counter() - start)*1000:.2f}ms"
        )

    

    def cleanup(self):
        """
        Publisher and Subscriber destructor
        """
        del self._dds_pcl_subscriber
        del self._dds_entities_publisher
        del self._dds_odometry_subscriber
        del self._dds_entities_subscriber


def main() -> None:
    server = ExampleDDS()
    signal.signal(signal.SIGINT, lambda *_: server.stop_request.set())
    server.run()    


if __name__ == "__main__":
    main()