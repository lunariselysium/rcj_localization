from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def enabled_condition(name):
    return IfCondition(LaunchConfiguration(name))


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
            DeclareLaunchArgument("enable_openmv_camera", default_value="true"),
            DeclareLaunchArgument("enable_fastmap_remap", default_value="true"),
            DeclareLaunchArgument("enable_black_feature_detector", default_value="true"),
            DeclareLaunchArgument("camera_enable_image_view", default_value="false"),
            DeclareLaunchArgument("camera_show_published_image", default_value="true"),
            DeclareLaunchArgument("remap_enable_image_view", default_value="false"),
            DeclareLaunchArgument("remap_show_input_image", default_value="false"),
            DeclareLaunchArgument("remap_show_output_image", default_value="true"),
            DeclareLaunchArgument("detector_enable_image_view", default_value="false"),
            DeclareLaunchArgument("detector_show_input_image", default_value="false"),
            DeclareLaunchArgument("detector_show_debug_image", default_value="true"),
            Node(
                package="rcj_localization",
                executable="image_publisher.py",
                name="openmv_camera",
                output="screen",
                condition=enabled_condition("enable_openmv_camera"),
                parameters=[
                    {
                        "port": LaunchConfiguration("port"),
                        "baudrate": LaunchConfiguration("baudrate"),
                        "timeout_sec": LaunchConfiguration("timeout_sec"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "image_topic": LaunchConfiguration("input_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                        "enable_image_view": LaunchConfiguration("camera_enable_image_view"),
                        "show_published_image": LaunchConfiguration("camera_show_published_image"),
                    }
                ],
            ),
            Node(
                package="rcj_localization",
                executable="fastmap_remap_node",
                name="black_feature_input_remap_node",
                output="screen",
                condition=enabled_condition("enable_fastmap_remap"),
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
            Node(
                package="rcj_localization",
                executable="black_feature_detector_node",
                name="black_feature_detector_node",
                output="screen",
                condition=enabled_condition("enable_black_feature_detector"),
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("output_topic"),
                        "enable_image_view": LaunchConfiguration("detector_enable_image_view"),
                        "show_input_image": LaunchConfiguration("detector_show_input_image"),
                        "show_debug_image": LaunchConfiguration("detector_show_debug_image"),
                        "use_v_aux_gate": False,
                    }
                ],
            ),
        ]
    )
