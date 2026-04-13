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
            "undistort_map_20260413_115400_fast.xml",
        ]
    )

    port = LaunchConfiguration("port")
    baudrate = LaunchConfiguration("baudrate")
    timeout_sec = LaunchConfiguration("timeout_sec")
    frame_id = LaunchConfiguration("frame_id")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    input_topic = LaunchConfiguration("input_topic")
    fastmap_file = LaunchConfiguration("fastmap_file")
    input_transport = LaunchConfiguration("input_transport")
    interpolation = LaunchConfiguration("interpolation")

    morph_node_name = "white_line_lab_morph_node"

    remap_specs = [
        (
            "white_morph_mask",
            f"/{morph_node_name}/white_morph_mask",
            "/white_line_lab_remap/white_morph_mask",
            "white_line_lab_remap_white_morph_mask_node",
        ),
        (
            "green_mask",
            f"/{morph_node_name}/green_mask",
            "/white_line_lab_remap/green_mask",
            "white_line_lab_remap_green_mask_node",
        ),
        (
            "black_mask",
            f"/{morph_node_name}/black_mask",
            "/white_line_lab_remap/black_mask",
            "white_line_lab_remap_black_mask_node",
        ),
        (
            "noise_mask",
            f"/{morph_node_name}/noise_mask",
            "/white_line_lab_remap/noise_mask",
            "white_line_lab_remap_noise_mask_node",
        ),
    ]

    remap_nodes = [
        Node(
            package="rcj_localization",
            executable="fastmap_remap_node",
            name=node_name,
            output="screen",
            parameters=[
                {
                    "fastmap_file": fastmap_file,
                    "input_topic": source_topic,
                    "output_topic": remapped_topic,
                    "input_transport": input_transport,
                    "interpolation": interpolation,
                }
            ],
        )
        for _, source_topic, remapped_topic, node_name in remap_specs
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("baudrate", default_value="115200"),
            DeclareLaunchArgument("timeout_sec", default_value="1.0"),
            DeclareLaunchArgument("frame_id", default_value="camera"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("input_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("fastmap_file", default_value=fastmap_default),
            DeclareLaunchArgument("input_transport", default_value="raw"),
            DeclareLaunchArgument("interpolation", default_value="nearest"),
            Node(
                package="rcj_localization",
                executable="image_publisher.py",
                name="openmv_camera",
                output="screen",
                parameters=[
                    {
                        "port": port,
                        "baudrate": baudrate,
                        "timeout_sec": timeout_sec,
                        "frame_id": frame_id,
                        "image_topic": input_topic,
                        "camera_info_topic": camera_info_topic,
                    }
                ],
            ),
            Node(
                package="rcj_localization",
                executable="white_line_lab_morph_node",
                name=morph_node_name,
                output="screen",
                parameters=[
                    {
                        "input_topic": input_topic,
                        "green_a_max": 117,
                        "enable_image_view": True,
                    }
                ],
            ),
            *remap_nodes,
            Node(
                package="rcj_localization",
                executable="white_line_skeleton_filter_node",
                name="white_line_skeleton_filter_node",
                output="screen",
                parameters=[
                    {
                        "morph_mask_topic": "/white_line_lab_remap/white_morph_mask",
                        "green_mask_topic": "/white_line_lab_remap/green_mask",
                        "black_mask_topic": "/white_line_lab_remap/black_mask",
                        "noise_mask_topic": "/white_line_lab_remap/noise_mask",
                        "orientation_window_radius_px": 5,
                        "min_orientation_neighbors": 6,
                        "enable_image_view": False,
                    }
                ],
            ),
        ]
    )
