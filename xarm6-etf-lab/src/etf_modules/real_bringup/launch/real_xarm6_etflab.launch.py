#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2022, ETFSA
# All rights reserved.
#
# Author: Dinko Osmankovic <dinko.osmankovic@etf.unsa.ba>


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip', default='192.168.1.236')
    report_type = LaunchConfiguration('report_type', default='normal')     # normal, rich, dev (see: https://github.com/xArm-Developer/xarm_ros#report_type-argument)
    dof = LaunchConfiguration('dof', default='6')
    prefix = LaunchConfiguration('prefix', default='')
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')
    limited = LaunchConfiguration('limited', default=True)
    effort_control = LaunchConfiguration('effort_control', default=False)
    velocity_control = LaunchConfiguration('velocity_control', default=False)
    add_gripper = LaunchConfiguration('add_gripper', default=True)
    add_vacuum_gripper = LaunchConfiguration('add_vacuum_gripper', default=False)
    baud_checkset = LaunchConfiguration('baud_checkset', default=True)
    default_gripper_baud = LaunchConfiguration('default_gripper_baud', default=2000000)
    robot_type = LaunchConfiguration('robot_type', default='xarm')

    # robot moveit servo launch
    # xarm_moveit_servo/launch/_robot_moveit_servo.launch.py
    robot_moveit_servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_servo'), 'launch', '_robot_moveit_servo.launch.py'])),
        launch_arguments={
            'robot_ip': robot_ip,
            'report_type': report_type,
            'baud_checkset': baud_checkset,
            'default_gripper_baud': default_gripper_baud,
            'dof': dof,
            'prefix': prefix,
            'hw_ns': hw_ns,
            'limited': limited,
            'effort_control': effort_control,
            'velocity_control': velocity_control,
            'add_gripper': add_gripper,
            'add_vacuum_gripper': add_vacuum_gripper,
            'robot_type': robot_type,
            'ros2_control_plugin': 'uf_robot_hardware/UFRobotSystemHardware',
        }.items(),
    )

    # robot driver launch
    # xarm_api/launch/_robot_driver.launch.py
    # This file calls xarm_api/config/xarm_user_params.yaml, so check what services are enabled/disabled
    robot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_api'), 'launch', '_robot_driver.launch.py'])),
        launch_arguments={
            'robot_ip': robot_ip,
            'report_type': report_type,
            'dof': dof,
            'hw_ns': hw_ns,
            'add_gripper': add_gripper,
            'prefix': prefix,
            'baud_checkset': baud_checkset,
            'default_gripper_baud': default_gripper_baud,
            'robot_type': robot_type,
        }.items(),
    )

    tf_node_world_base = Node(package = "tf2_ros", 
                   executable = "static_transform_publisher",
                   arguments = ["0", "0", "0.5", "0", "0", "1.571", "world", "link_base"]
				)

    return LaunchDescription([
        robot_moveit_servo_launch,
        tf_node_world_base,
        robot_driver_launch
    ])
