from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def find_latest_fastmap_file():
    config_dir = Path(get_package_share_directory("rcj_localization")) / "config"
    candidates = sorted(config_dir.glob("undistort_map_*_fast.xml"))
    if not candidates:
        candidates = sorted(config_dir.glob("*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No fastmap XML file found in {config_dir}")
    return candidates[-1]


def read_fastmap_source_size(fastmap_file):
    root = ET.parse(fastmap_file).getroot()
    source_width = root.findtext("source_width")
    source_height = root.findtext("source_height")
    if source_width is None or source_height is None:
        raise RuntimeError(
            f"Fastmap XML '{fastmap_file}' is missing source_width/source_height"
        )
    return int(source_width), int(source_height)


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
    fastmap_default = find_latest_fastmap_file()
    default_width, default_height = read_fastmap_source_size(fastmap_default)

    camera_index = LaunchConfiguration("camera_index")
    role = LaunchConfiguration("role")
    image_format = LaunchConfiguration("format")
    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    orientation = LaunchConfiguration("orientation")
    frame_id = LaunchConfiguration("frame_id")
    camera_info_url = LaunchConfiguration("camera_info_url")
    use_node_time = LaunchConfiguration("use_node_time")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    fastmap_file = LaunchConfiguration("fastmap_file")
    input_transport = LaunchConfiguration("input_transport")
    interpolation = LaunchConfiguration("interpolation")

    morph_node_name = "white_line_lab_morph_node"
    skeleton_show_white_final_mask = resolve_legacy_bool(
        "skeleton_show_white_final_mask",
        "skeleton_show_white_mask",
    )

    launch_arguments = [
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("role", default_value="viewfinder"),
        DeclareLaunchArgument("format", default_value="RGB888"),
        DeclareLaunchArgument("width", default_value=str(default_width)),
        DeclareLaunchArgument("height", default_value=str(default_height)),
        DeclareLaunchArgument("orientation", default_value="0"),
        DeclareLaunchArgument("frame_id", default_value="camera"),
        DeclareLaunchArgument("camera_info_url", default_value=""),
        DeclareLaunchArgument("use_node_time", default_value="false"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("input_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("output_topic", default_value="/camera/image_remapped"),
        DeclareLaunchArgument("fastmap_file", default_value=str(fastmap_default)),
        DeclareLaunchArgument("input_transport", default_value="raw"),
        DeclareLaunchArgument("interpolation", default_value="linear"),
        DeclareLaunchArgument("camera_enable_image_view", default_value="false"),
        DeclareLaunchArgument("camera_show_published_image", default_value="true"),
        DeclareLaunchArgument("remap_enable_image_view", default_value="false"),
        DeclareLaunchArgument("remap_show_input_image", default_value="false"),
        DeclareLaunchArgument("remap_show_output_image", default_value="true"),
        DeclareLaunchArgument("morph_enable_image_view", default_value="false"),
        DeclareLaunchArgument("morph_show_input_image", default_value="false"),
        DeclareLaunchArgument("morph_show_white_candidate_mask", default_value="true"),
        DeclareLaunchArgument("morph_show_white_morph_mask", default_value="true"),
        DeclareLaunchArgument("morph_show_green_mask", default_value="false"),
        DeclareLaunchArgument("morph_show_black_mask", default_value="false"),
        DeclareLaunchArgument("morph_show_noise_mask", default_value="false"),
        DeclareLaunchArgument("morph_show_debug_image", default_value="false"),
        DeclareLaunchArgument("skeleton_enable_image_view", default_value="false"),
        DeclareLaunchArgument("skeleton_show_morph_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_green_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_black_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_noise_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_skeleton_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_orientation_valid_mask", default_value="false"),
        DeclareLaunchArgument("skeleton_show_side_support_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_width_supported_skeleton_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_supported_skeleton_mask", default_value="true"),
        DeclareLaunchArgument(
            "skeleton_show_length_filtered_skeleton_mask",
            default_value=LaunchConfiguration("skeleton_show_supported_skeleton_mask"),
        ),
        DeclareLaunchArgument("skeleton_show_reconstructed_mask", default_value="true"),
        DeclareLaunchArgument("skeleton_show_white_final_mask", default_value="true"),
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
                        "camera": ParameterValue(camera_index, value_type=int),
                        "role": role,
                        "format": image_format,
                        "width": ParameterValue(width, value_type=int),
                        "height": ParameterValue(height, value_type=int),
                        "orientation": ParameterValue(orientation, value_type=int),
                        "frame_id": frame_id,
                        "camera_info_url": camera_info_url,
                        "use_node_time": ParameterValue(use_node_time, value_type=bool),
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
        ]
    )
