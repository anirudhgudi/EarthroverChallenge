import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mango/Earthrover/earth-rovers-sdk/examples/ros2/zero_shot_ws/install/zero_shot_navigation'
