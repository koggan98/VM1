from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tracking_pkg',
            executable='camera_publisher.py', 
            name='camera_publisher',
            output='screen'
        ),
        Node(
            package='tracking_pkg',
            executable='frame_publisher.py',
            name='frame_publisher',
            output='screen'
        ),
        Node(
            package='tracking_pkg',
            executable='hand_pose_visualization_publisher.py', 
            name='hand_pose_visualization_publisher',
            output='screen'
        ),
        Node(
            package='tracking_pkg',
            executable='hand_tracker.py', 
            name='hand_tracker',
            output='screen'
        ),
 #       Node(
  #          package='tracking_pkg',
   #         executable='box_publisher.py', 
    #        name='box_publisher',
     #       output='screen'
      #  ),
        Node(
            package='tracking_pkg',
            executable='gripper_mover.py', 
            name='gripper_mover',
            output='screen'
        ),
        
        # Verzögerter Start der Node um 5 Sekunden
        TimerAction(
            period=5.0,  # Verzögerung in Sekunden
            actions=[
                Node(
                package='tracking_pkg',
                executable='moveit_mover', 
                name='moveit_mover',
                output='screen'
                )
            ]
        )
    ])
