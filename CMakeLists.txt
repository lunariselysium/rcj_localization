cmake_minimum_required(VERSION 3.8)
project(rcj_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
# Crucially, find the OpenCV package that cv_bridge depends on
find_package(OpenCV REQUIRED)

# --- Add Your Executable ---
# This tells cmake to build an executable file named "localization_node"
# from your source file "src/main_node.cpp"
add_executable(localization_node src/main_node.cpp)

# --- Link Libraries ---
# This is the most important part. It connects your code to the ROS 2 and OpenCV libraries.
ament_target_dependencies(localization_node
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
  "geometry_msgs"
  "nav_msgs"
  "tf2_ros"
  "cv_bridge"
  "visualization_msgs"
)

# Explicitly link OpenCV libraries to your target
target_link_libraries(localization_node ${OpenCV_LIBRARIES})


# --- Installation Rules ---
# This tells colcon where to put the files so `ros2 run` can find them.
install(TARGETS
  localization_node
  DESTINATION lib/${PROJECT_NAME}
)

# --- Install Python Scripts ---
install(PROGRAMS
  "scripts/yaw_publisher.py"
  DESTINATION lib/${PROJECT_NAME}
)

# --- HSV Tuner ---
add_executable(hsv_tuner src/hsv_tuner_node.cpp)
ament_target_dependencies(hsv_tuner
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)
install(TARGETS
  hsv_tuner
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
