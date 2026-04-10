import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import serial
import numpy as np
import cv2

class OpenMVPublisher(Node):
    def __init__(self):
        super().__init__('openmv_camera')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.bridge = CvBridge()

        # 打开串口，连接到 OpenMV
        self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=1)  # 更高的波特率
        self.timer = self.create_timer(0.1, self.publish_frame)  # 每0.1秒发布一帧

    def read_exact(self, n):
        data = b''
        while len(data) < n:
            packet = self.ser.read(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def read_frame(self):
        while True:
            header = self.ser.read(2)  # 读取帧头
            if header == b'\xAA\xBB':  # 检查是否是有效帧头
                width = int.from_bytes(self.read_exact(2), 'little')
                height = int.from_bytes(self.read_exact(2), 'little')
                length = int.from_bytes(self.read_exact(4), 'little')

                img_bytes = self.read_exact(length)
                if img_bytes is None:
                    return None

                # 解码JPEG图像
                img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    return None

                return img

    def publish_frame(self):
        frame = self.read_frame()  # 获取一帧图像

        if frame is None:
            self.get_logger().warn("No frame")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')  # 将图像转换为 ROS 消息
        msg.header.frame_id = "camera"
        msg.header.stamp = self.get_clock().now().to_msg()

        # 发布相机信息
        camera_info = CameraInfo()
        camera_info.header.stamp = msg.header.stamp
        camera_info.header.frame_id = "camera"
        camera_info.width = frame.shape[1]
        camera_info.height = frame.shape[0]
        camera_info.distortion_model = "plumb_bob"
        camera_info.k = [0.0] * 9
        camera_info.d = [0.0] * 4
        camera_info.r = [0.0] * 9
        camera_info.p = [0.0] * 12

        self.info_pub.publish(camera_info)
        self.pub.publish(msg)  # 发布图像消息

def main():
    rclpy.init()
    node = OpenMVPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()