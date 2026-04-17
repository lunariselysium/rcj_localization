from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("rcj_localization")
    fastmap_default = PathJoinSubstitution(
        [package_share, "config", "undistort_map_20260414_204537_fast.xml"]
    )
    pipeline_launch = PathJoinSubstitution(
        [
            package_share,
            "launch",
            "white_line_lab_skeleton_fastmap_localization.launch.py",
        ]
    )

    input_topic = LaunchConfiguration("input_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    output_topic = LaunchConfiguration("output_topic")
    fastmap_file = LaunchConfiguration("fastmap_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_index", default_value="0"),
            DeclareLaunchArgument("role", default_value="viewfinder"),
            DeclareLaunchArgument("format", default_value="RGB888"),
            DeclareLaunchArgument("width", default_value="800"),
            DeclareLaunchArgument("height", default_value="600"),
            DeclareLaunchArgument("orientation", default_value="0"),
            DeclareLaunchArgument("frame_id", default_value="camera"),
            DeclareLaunchArgument("camera_info_url", default_value=""),
            DeclareLaunchArgument("use_node_time", default_value="false"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("input_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("output_topic", default_value="/camera/image_remapped"),
            DeclareLaunchArgument("fastmap_file", default_value=fastmap_default),
            Node(
                package="camera_ros",
                executable="camera_node",
                name="camera",
                output="screen",
                remappings=[
                    ("~/image_raw", input_topic),
                    ("~/camera_info", camera_info_topic),
                ],
                parameters=[
                    {
                        "camera": ParameterValue(
                            LaunchConfiguration("camera_index"),
                            value_type=int,
                        ),
                        "role": LaunchConfiguration("role"),
                        "format": LaunchConfiguration("format"),
                        "width": ParameterValue(LaunchConfiguration("width"), value_type=int),
                        "height": ParameterValue(LaunchConfiguration("height"), value_type=int),
                        "orientation": ParameterValue(
                            LaunchConfiguration("orientation"),
                            value_type=int,
                        ),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "camera_info_url": LaunchConfiguration("camera_info_url"),
                        "use_node_time": ParameterValue(
                            LaunchConfiguration("use_node_time"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(pipeline_launch),
                launch_arguments={
                    "enable_openmv_camera": "false",
                    "camera_info_topic": camera_info_topic,
                    "input_topic": input_topic,
                    "output_topic": output_topic,
                    "fastmap_file": fastmap_file,
                }.items(),
            ),
        ]
    )
