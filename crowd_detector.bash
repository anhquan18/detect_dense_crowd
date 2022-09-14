#!/bin/bash

sudo chmod 666 /dev/ttyACM0
roslaunch sakuron_bringup arduino_driver.launch
roslaunch crowd_detection crowd_detector.launch
rosrun sakuron_apps following
roslaunch sakuron_bringup motor.launch
cd ~/EPOSx_2wheels/build/
./epos_2wheels --yaml_file ../yaml/cfg_EPOS.yaml --wheel_spd_path ../yaml/
