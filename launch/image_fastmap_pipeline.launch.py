from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    fastmap_default = PathJoinSubstitution(
        [
            FindPackageShare("rcj_localization"),
            "config",
            "undistort_map_20260414_204537_fast.xml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("baudrate", default_value="115200"),
            DeclareLaunchArgument("timeout_sec", default_value="1.0"),
            DeclareLaunchArgument("frame_id", default_value="camera"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("input_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("output_topic", default_value="/camera/image_remapped"),
            DeclareLaunchArgument("fastmap_file", default_value=fastmap_default),
            DeclareLaunchArgument("input_transport", default_value="raw"),
            DeclareLaunchArgument("interpolation", default_value="linear"),
            DeclareLaunchArgument("remap_enable_image_view", default_value="true"),
            DeclareLaunchArgument("remap_show_input_image", default_value="true"),
            DeclareLaunchArgument("remap_show_output_image", default_value="true"),
            Node(
                package="rcj_localization",
                executable="image_publisher.py",
                name="openmv_camera",
                output="screen",
                parameters=[
                    {
                        "port": LaunchConfiguration("port"),
                        "baudrate": LaunchConfiguration("baudrate"),
                        "timeout_sec": LaunchConfiguration("timeout_sec"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "image_topic": LaunchConfiguration("input_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                    }
                ],
            ),
            Node(
                package="rcj_localization",
                executable="fastmap_remap_node",
                name="fastmap_remap_node",
                output="screen",
                parameters=[
                    {
                        "fastmap_file": LaunchConfiguration("fastmap_file"),
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "input_transport": LaunchConfiguration("input_transport"),
                        "interpolation": LaunchConfiguration("interpolation"),
                        "enable_image_view": LaunchConfiguration("remap_enable_image_view"),
                        "show_input_image": LaunchConfiguration("remap_show_input_image"),
                        "show_output_image": LaunchConfiguration("remap_show_output_image"),
                    }
                ],
            ),
        ]
    )
