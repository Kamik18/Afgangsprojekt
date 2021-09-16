import subprocess
import sys

def install_package(package):
    print("Installing: " + package)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_package("matplotlib")
install_package("numpy")
install_package("scipy")
install_package("pygame")