import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_tuner_package_name = 'rcj_localization' 
    
    # Step 2: (Optional) Path to the map file to display in RViz
    # This part is kept from your example to publish the map.
    map_file_path = os.path.join(
        get_package_share_directory('rcj_localization'),
        'maps',
        'rcj_map.yaml'
    )

    return LaunchDescription([
        # --- MAP PUBLISHING (Kept from your example) ---
        # Starts the Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file_path}]
        ),

        # Starts the Lifecycle Manager to activate the Map Server
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_for_map', # Renamed to avoid conflicts
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server']
            }]
        ),
        
        # --- CAMERA TUNER NODE ---
        # Starts your actual camera tuner node
        Node(
            package=camera_tuner_package_name,
            executable='camera_tuner', # The name of your C++ executable
            name='camera_tuner',
            output='screen'
            # No parameters are needed for this node
        ),
    ])