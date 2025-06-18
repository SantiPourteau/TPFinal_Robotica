#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import transforms3d as tf_transformations
# from tf2_ros import transformations as tft
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        # OK TODO: define map resolution, width, height, and number of particles
        self.declare_parameter('map_resolution', 0.05)  # metros por celda
        self.declare_parameter('map_width_meters', 10.0)  # ancho en metros
        self.declare_parameter('map_height_meters', 10.0)  # alto en metros
        self.declare_parameter('num_particles', 100)  # número de partículas

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # OK TODO: define the log-odds criteria for free and occupied cells
        self.log_odds_occupied = 0.85
        self.log_odds_free = -0.4
        self.log_odds_max = 5.0
        self.log_odds_min = -5.0

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # OK TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

        quaternion = [orientation.w, orientation.x, orientation.y, orientation.z]  # ¡Atención! orden: w, x, y, z
        _, _, yaw = tf_transformations.euler.quat2euler(quaternion)

        odom_pose = (position.x, position.y, yaw)

        # TODO: Model the particles around the current pose

        if not hasattr(self, 'previous_odom_pose'):
            self.previous_odom_pose = odom_pose
            return
        
        dx = odom_pose[0] - self.previous_odom_pose[0]
        dy = odom_pose[1] - self.previous_odom_pose[1]
        dtheta = self.angle_diff(odom_pose[2], self.previous_odom_pose[2]) #np.arctan2(np.sin(a-b), np.cos(a-b))

        for p in self.particles:
            noisy_dx = dx + np.random.normal(0, 0.01)
            noisy_dy = dy + np.random.normal(0, 0.01)
            noisy_dtheta = dtheta + np.random.normal(0, 0.01)

            p.x += noisy_dx * np.cos(p.theta) - noisy_dy * np.sin(p.theta)
            p.y += noisy_dx * np.sin(p.theta) + noisy_dy * np.cos(p.theta)
            p.theta += noisy_dtheta

        self.previous_odom_pose = odom_pose

        # OK TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            weights.append(weight)
            p.weight = weight # Esto es el save??

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Si todos los pesos son cero, asigna pesos uniformes
            weights = [1.0 / self.num_particles] * self.num_particles

        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # OK TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        mean_x = sum(p.x * p.weight for p in self.particles)
        mean_y = sum(p.y * p.weight for p in self.particles)

        # Para el ángulo:
        mean_sin = sum(np.sin(p.theta) * p.weight for p in self.particles)
        mean_cos = sum(np.cos(p.theta) * p.weight for p in self.particles)
        mean_theta = math.atan2(mean_sin, mean_cos)

        # Guardamos la pose estimada
        self.current_map_pose = (mean_x, mean_y, mean_theta)
        self.current_odom_pose = odom_pose  # Odom pose es el ultimo odom message 

        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue

            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            end_x = robot_x + range_dist * math.cos(robot_theta + angle)
            end_y = robot_y + range_dist * math.sin(robot_theta + angle)
            cell_x = int((end_x - self.map_origin_x) / self.resolution)
            cell_y = int((end_y - self.map_origin_y) / self.resolution)

            # TODO: Use particle.log_odds_map for scoring
            if 0 <= cell_x < self.map_width_cells and 0 <= cell_y < self.map_height_cells:
                if particle.log_odds_map[cell_y, cell_x] > 0:
                    score += 1.0
        return score + 1e-6

    def resample_particles(self, particles):
        # TODO: Resample particles
        new_particles = []
        weights = [p.weight for p in particles]
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # evitar errores por redondeo

        step = 1.0 / self.num_particles
        r = np.random.uniform(0, step)
        indexes = []
        i = 0

        for m in range(self.num_particles):
            u = r + m * step
            while u > cumulative_sum[i]:
                i += 1
            indexes.append(i)

        for idx in indexes:
            p = particles[idx]
            new_p = Particle(p.x, p.y, p.theta, 1.0 / self.num_particles, p.log_odds_map.shape)
            new_p.log_odds_map = np.copy(p.log_odds_map)
            new_particles.append(new_p)

        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
            # OK TODO: Update map: transform the scan into the map frame
            # OK TODO: Use self.bresenham_line for free cells
            # OK TODO: Update particle.log_odds_map accordingly

            # --- 1. Transformar el rayo al marco del mapa
        angle = scan_msg.angle_min + i * scan_msg.angle_increment
        angle_world = robot_theta + angle

        hit_x = robot_x + current_range * math.cos(angle_world)
        hit_y = robot_y + current_range * math.sin(angle_world)

        # Convertir a coordenadas del mapa (índices de celda)
        map_x0 = int((robot_x - self.map_origin_x) / self.resolution) # posicion del robot
        map_y0 = int((robot_y - self.map_origin_y) / self.resolution) # posicion del robot
        map_x1 = int((hit_x - self.map_origin_x) / self.resolution) # posicion del hit
        map_y1 = int((hit_y - self.map_origin_y) / self.resolution) # posicion del hit

        # --- 2. Raytrace celdas libres
        self.bresenham_line(particle, map_x0, map_y0, map_x1, map_y1)

        # --- 3. Actualizar celda de impacto como ocupada (si es hit)
        if is_hit:
            if (0 <= map_x1 < self.map_width_cells) and (0 <= map_y1 < self.map_height_cells):
                particle.log_odds_map[map_y1, map_x1] += self.log_odds_occupied
                particle.log_odds_map[map_y1, map_x1] = np.clip(
                    particle.log_odds_map[map_y1, map_x1], self.log_odds_min, self.log_odds_max
                )

            

    def bresenham_line(self, particle, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # OK TODO: Fill in map_msg fields and publish one map
        best_particle = max(self.particles, key=lambda p: p.weight)

        # Crear mensaje OccupancyGrid
        map_msg = OccupancyGrid()

        # Header
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        # Info del mapa
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0  # sin rotación

        # Convertir log-odds a formato de ROS (0 = libre, 100 = ocupado, -1 = desconocido)
        # Usando operaciones vectorizadas de NumPy para mejor rendimiento
        log_odds = best_particle.log_odds_map
        data = np.zeros_like(log_odds, dtype=np.int8)
        
        # Desconocido (exactamente 0.0)
        data[log_odds == 0.0] = -1
        
        # Ocupado (positivo)
        occupied_mask = log_odds > 0.0
        data[occupied_mask] = np.minimum(100, (100 * log_odds[occupied_mask] / self.log_odds_max)).astype(np.int8)
        
        # Libre (negativo)
        free_mask = log_odds < 0.0
        data[free_mask] = np.maximum(0, (100 * log_odds[free_mask] / self.log_odds_min)).astype(np.int8)

        map_msg.data = data.flatten().tolist()
        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def broadcast_map_to_odom(self):
        # OK TODO: Broadcast map->odom transform
        if not hasattr(self, 'current_map_pose'):
            return

        x, y, theta = self.current_map_pose

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()