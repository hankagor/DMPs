add_executable(test_move_xarm6 test_move_xarm6.cpp)
target_compile_features(test_move_xarm6 PUBLIC c_std_99 cxx_std_17)
target_link_libraries(test_move_xarm6 PUBLIC sim_bringup)

add_executable(test_read_octomap test_read_octomap.cpp)
target_compile_features(test_read_octomap PUBLIC c_std_99 cxx_std_17)
target_link_libraries(test_read_octomap PUBLIC sim_bringup)

add_executable(test_planning test_planning.cpp)
target_compile_features(test_planning PUBLIC c_std_99 cxx_std_17)
target_link_libraries(test_planning PUBLIC sim_bringup)

add_executable(test_task_planning test_task_planning.cpp)
target_compile_features(test_task_planning PUBLIC c_std_99 cxx_std_17)
target_link_libraries(test_task_planning PUBLIC sim_bringup)

add_executable(test_real_time_planning test_real_time_planning.cpp)
target_compile_features(test_real_time_planning PUBLIC c_std_99 cxx_std_17)
target_link_libraries(test_real_time_planning PUBLIC sim_bringup)

add_executable(test_draw test_draw.cpp)
target_compile_features(test_draw PUBLIC c_std_99 cxx_std_17)
target_link_libraries(test_draw PUBLIC sim_bringup yaml-cpp)
install(TARGETS 
  test_move_xarm6
  test_read_octomap
  test_planning
  test_task_planning
  test_real_time_planning
  test_draw
  DESTINATION lib/${PROJECT_NAME})
