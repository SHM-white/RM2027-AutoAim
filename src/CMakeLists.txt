# Keep executable paths compatible with deployment scripts: build/<target>.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(standard_mpc standard_mpc.cpp)
target_link_libraries(standard_mpc PRIVATE
  ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)

add_executable(auto_aim_debug_mpc auto_aim_debug_mpc.cpp)
target_link_libraries(auto_aim_debug_mpc PRIVATE
  ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(auto_aim_debug auto_aim_debug.cpp)
target_link_libraries(auto_aim_debug PRIVATE
  ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(auto_buff_debug_mpc auto_buff_debug_mpc.cpp)
target_link_libraries(auto_buff_debug_mpc PRIVATE
  ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io)

if(SPVISION_WITH_ROS2)
  add_executable(sentry sentry.cpp)
  target_link_libraries(sentry PRIVATE
    ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception sentry_decision tools io)
endif()
