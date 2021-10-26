import subprocess
import sys

def install_package(package):
    print("Installing: " + package)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_package("matplotlib")
install_package("numpy")
install_package("scipy")
install_package("pygame")
install_package("opencv-python")
install_package("pyrealsense2")
install_package("tensorflow==2.5.0")
install_package("--use-deprecated=legacy-resolver tflite-model-maker")
install_package("pycocotools")