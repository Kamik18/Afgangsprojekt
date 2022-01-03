#!/bin/bash

gpio export 4 out
sudo sh -c "echo 0 > /sys/class/gpio/gpio4/value"
sleep 0.5
sudo sh -c "echo 1 > /sys/class/gpio/gpio4/value"
