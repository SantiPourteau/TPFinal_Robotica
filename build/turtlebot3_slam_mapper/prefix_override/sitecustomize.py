import sys
if sys.prefix == '/Users/Colegio/miniforge3/envs/rosenv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/Users/Colegio/Documents/4to/1er_Semestre/Robotica/TPFinal_Robotica/install/turtlebot3_slam_mapper'
