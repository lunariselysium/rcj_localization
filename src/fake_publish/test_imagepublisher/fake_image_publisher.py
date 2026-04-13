#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


IMAGE_PATTERNS = ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.webp")
PUBLISH_TOPIC = "/camera/image_raw"
PUBLISH_INTERVAL_SEC = 3
FRAME_ID = "camera"


def find_images(image_dir):
    image_paths = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(image_dir.glob(pattern))
    return sorted(path for path in image_paths if path.is_file())


def main():
    image_dir = Path(__file__).resolve().parent
    image_paths = find_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    rclpy.init()
    node = rclpy.create_node("fake_image_publisher")
    publisher = node.create_publisher(Image, PUBLISH_TOPIC, 10)
    bridge = CvBridge()

    node.get_logger().info(
        f"Publishing {len(image_paths)} image(s) from {image_dir} to {PUBLISH_TOPIC}"
    )

    try:
        while rclpy.ok():
            for image_path in image_paths:
                if not rclpy.ok():
                    break

                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    node.get_logger().warning(f"Failed to read image: {image_path.name}")
                    continue

                msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = FRAME_ID
                publisher.publish(msg)

                node.get_logger().info(f"Published {image_path.name}")
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(PUBLISH_INTERVAL_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
