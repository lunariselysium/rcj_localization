from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def resolve_legacy_bool(current_name, legacy_name):
    return ParameterValue(
        PythonExpression(
            [
                "'",
                LaunchConfiguration(legacy_name),
                "' != '__unset__' and '",
                LaunchConfiguration(legacy_name),
                "' or '",
                LaunchConfiguration(current_name),
                "'",
            ]
        ),
        value_type=bool,
    )


def legacy_arg_used_condition(name):
    return IfCondition(
        PythonExpression(["'", LaunchConfiguration(name), "' != '__unset__'"])
    )


def generate_launch_description():
    package_share = FindPackageShare("rcj_localization")
    fastmap_default = PathJoinSubstitution(
        [package_share, "config", "undistort_map_20260414_204537_fast.xml"]
    )
    map_yaml_default = PathJoinSubstitution([package_share, "maps", "rcj_map.yaml"])

    port = LaunchConfiguration("port")
    baudrate = LaunchConfiguration("baudrate")
    timeout_sec = LaunchConfiguration("timeout_sec")
    frame_id = LaunchConfiguration("frame_id")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    fastmap_file = LaunchConfiguration("fastmap_file")
    input_transport = LaunchConfiguration("input_transport")
    interpolation = LaunchConfiguration("interpolation")
    map_yaml_file = LaunchConfiguration("map_yaml_file")
    use_fake_yaw = LaunchConfiguration("use_fake_yaw")
    yaw_topic = LaunchConfiguration("yaw_topic")
    mask_topic = LaunchConfiguration("mask_topic")

    morph_node_name = "white_line_lab_morph_node"
    skeleton_show_white_final_mask = resolve_legacy_bool(
        "skeleton_show_white_final_mask",
        "skeleton_show_white_mask",
    )

    launch_arguments = [
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
        DeclareLaunchArgument("map_yaml_file", default_value=map_yaml_default),
        DeclareLaunchArgument("use_fake_yaw", default_value="true"),
        DeclareLaunchArgument("yaw_topic", default_value="/robot/yaw"),
        DeclareLaunchArgument(
            "mask_topic",
            default_value="/white_line_skeleton_filter_node/white_final_mask",
        ),
        DeclareLaunchArgument("meters_per_pixel", default_value="-1.0"),
        DeclareLaunchArgument("forward_axis", default_value="__unset__"),
        DeclareLaunchArgument("left_axis", default_value="__unset__"),
        DeclareLaunchArgument("max_points", default_value="5000"),
        DeclareLaunchArgument("num_particles", default_value="1000"),
        DeclareLaunchArgument("camera_enable_image_view", default_value="false"),
        DeclareLaunchArgument("camera_show_published_image", default_value="true"),
        DeclareLaunchArgument("remap_enable_image_view", default_value="false"),
        DeclareLaunchArgument("remap_show_input_image", default_value="true"),
        DeclareLaunchArgument("remap_show_output_image", default_value="true"),
        DeclareLaunchArgument("morph_enable_image_view", default_value="false"),
        DeclareLaunchArgument("morph_show_input_image", default_value="false"),
        DeclareLaunchArgument("morph_show_white_candidate_mask", default_value="true"),
        DeclareLaunchArgument("morph_show_white_morph_mask", default_value="true"),
        DeclareLaunchArgument("morph_show_green_mask", default_value="true"),
        DeclareLaunchArgument("morph_show_black_mask", default_value="false"),
        DeclareLaunchArgument("morph_show_noise_mask", default_value="false"),
        DeclareLaunchArgument("morph_show_debug_image", default_value="false"),
        DeclareLaunchArgument("skeleton_enable_image_view", default_value="true"),
        DeclareLaunchArgument("skeleton_show_morph_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_green_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_black_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_noise_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_skeleton_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_orientation_valid_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_side_support_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_width_supported_skeleton_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_supported_skeleton_mask", default_value="false"),
        DeclareLaunchArgument(
            "skeleton_show_length_filtered_skeleton_mask",
            default_value=LaunchConfiguration("skeleton_show_supported_skeleton_mask"),
        ),
        DeclareLaunchArgument("skeleton_show_reconstructed_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_white_final_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_white_mask", default_value="__unset__"),
        DeclareLaunchArgument("skeleton_show_debug_image", default_value="false"),
    ]

    return LaunchDescription(
        launch_arguments
        + [
            LogInfo(
                condition=legacy_arg_used_condition("skeleton_show_white_mask"),
                msg=(
                    "Launch argument 'skeleton_show_white_mask' is deprecated. "
                    "Use 'skeleton_show_white_final_mask' instead."
                ),
            ),
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
                        "enable_image_view": LaunchConfiguration("camera_enable_image_view"),
                        "show_published_image": LaunchConfiguration("camera_show_published_image"),
                    }
                ],
            ),
            Node(
                package="rcj_localization",
                executable="fastmap_remap_node",
                name="white_line_lab_input_remap_node",
                output="screen",
                parameters=[
                    {
                        "fastmap_file": fastmap_file,
                        "input_topic": input_topic,
                        "output_topic": output_topic,
                        "input_transport": input_transport,
                        "interpolation": interpolation,
                        "enable_image_view": LaunchConfiguration("remap_enable_image_view"),
                        "show_input_image": LaunchConfiguration("remap_show_input_image"),
                        "show_output_image": LaunchConfiguration("remap_show_output_image"),
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
                        "input_topic": output_topic,
                        "green_a_max": 117,
                        "enable_image_view": LaunchConfiguration("morph_enable_image_view"),
                        "show_input_image": LaunchConfiguration("morph_show_input_image"),
                        "show_white_candidate_mask": LaunchConfiguration(
                            "morph_show_white_candidate_mask"
                        ),
                        "show_white_morph_mask": LaunchConfiguration("morph_show_white_morph_mask"),
                        "show_green_mask": LaunchConfiguration("morph_show_green_mask"),
                        "show_black_mask": LaunchConfiguration("morph_show_black_mask"),
                        "show_noise_mask": LaunchConfiguration("morph_show_noise_mask"),
                        "show_debug_image": LaunchConfiguration("morph_show_debug_image"),
                    }
                ],
            ),
            Node(
                package="rcj_localization",
                executable="white_line_skeleton_filter_node",
                name="white_line_skeleton_filter_node",
                output="screen",
                parameters=[
                    {
                        "morph_mask_topic": f"/{morph_node_name}/white_morph_mask",
                        "green_mask_topic": f"/{morph_node_name}/green_mask",
                        "black_mask_topic": f"/{morph_node_name}/black_mask",
                        "noise_mask_topic": f"/{morph_node_name}/noise_mask",
                        "orientation_window_radius_px": 5,
                        "min_orientation_neighbors": 6,
                        "enable_image_view": LaunchConfiguration("skeleton_enable_image_view"),
                        "show_morph_mask": LaunchConfiguration("skeleton_show_morph_mask"),
                        "show_green_mask": LaunchConfiguration("skeleton_show_green_mask"),
                        "show_black_mask": LaunchConfiguration("skeleton_show_black_mask"),
                        "show_noise_mask": LaunchConfiguration("skeleton_show_noise_mask"),
                        "show_skeleton_mask": LaunchConfiguration("skeleton_show_skeleton_mask"),
                        "show_orientation_valid_mask": LaunchConfiguration(
                            "skeleton_show_orientation_valid_mask"
                        ),
                        "show_side_support_mask": LaunchConfiguration(
                            "skeleton_show_side_support_mask"
                        ),
                        "show_width_supported_skeleton_mask": LaunchConfiguration(
                            "skeleton_show_width_supported_skeleton_mask"
                        ),
                        "show_length_filtered_skeleton_mask": LaunchConfiguration(
                            "skeleton_show_length_filtered_skeleton_mask"
                        ),
                        "show_supported_skeleton_mask": LaunchConfiguration(
                            "skeleton_show_length_filtered_skeleton_mask"
                        ),
                        "show_reconstructed_mask": LaunchConfiguration(
                            "skeleton_show_reconstructed_mask"
                        ),
                        "show_white_final_mask": skeleton_show_white_final_mask,
                        "show_debug_image": LaunchConfiguration("skeleton_show_debug_image"),
                    }
                ],
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[{"yaml_filename": map_yaml_file}],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_localization",
                output="screen",
                parameters=[{"autostart": True, "node_names": ["map_server"]}],
            ),
            Node(
                package="rcj_localization",
                executable="yaw_publisher.py",
                name="yaw_publisher",
                output="screen",
                condition=IfCondition(use_fake_yaw),
                remappings=[("/robot/yaw", yaw_topic)],
            ),
            Node(
                package="rcj_localization",
                executable="topdown_mask_points_node",
                name="topdown_mask_points_node",
                output="screen",
                parameters=[
                    {
                        "mask_topic": mask_topic,
                        "output_topic": "/field_line_observations",
                        "meters_per_pixel": ParameterValue(
                            LaunchConfiguration("meters_per_pixel"),
                            value_type=float,
                        ),
                        "forward_axis": LaunchConfiguration("forward_axis"),
                        "left_axis": LaunchConfiguration("left_axis"),
                        "max_points": ParameterValue(
                            LaunchConfiguration("max_points"),
                            value_type=int,
                        ),
                    }
                ],
            ),
            Node(
                package="rcj_localization",
                executable="pf_localization_node",
                name="pf_localization_node",
                output="screen",
                parameters=[
                    {
                        "num_particles": ParameterValue(
                            LaunchConfiguration("num_particles"),
                            value_type=int,
                        ),
                        "map_topic": "/map",
                        "yaw_topic": yaw_topic,
                        "observations_topic": "/field_line_observations",
                    }
                ],
            ),
        ]
    )
