import socket
import numpy as np
import time
import threading
# PW ist "runrobot" !!


from ur_feedback_socket import URFeedback

class URCommand:
    def __init__(self, robot_ip, robot_command_port, robot_feedback_port, gripper_port):
        self.robot_ip = robot_ip
        self.robot_command_port = robot_command_port
        self.robot_feedback_port = robot_feedback_port
        self.gripper_port = gripper_port
        self.socket_ur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_ur.connect((self.robot_ip, self.robot_command_port))
        self.socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_gripper.connect((self.robot_ip, self.gripper_port))
        self.ur_feedback = URFeedback(self.robot_ip, self.robot_feedback_port)
        self.ur_feedback.connect()
        
        # Start the thread to update the program status
        self.status_thread = threading.Thread(target=self.ur_feedback.update_program_status)
        self.status_thread.daemon = True
        self.status_thread.start()

        print('UR Node started.')

    def ur_command(self, command):
        full_command = f"def my_prog():\n{command}\nend\n"
        try:
            self.socket_ur.sendall(full_command.encode('utf-8'))
            print(f'Sent URScript command: {full_command}')
        except socket.error as e:
            print(f"Socket error: {e}")

    def gripper_command(self, command):
        self.socket_gripper.sendall(command.encode('utf-8'))
        print(f'Sent Gripper command: {command}')

    def command_position(self, position, orientation, acceleration=1.5, speed=0.25, radius=0):
        position = np.array(position) / 1000
        script_command = f"movej(p[{position[0]}, {position[1]}, {position[2]}, {orientation[0]}, {orientation[1]}, {orientation[2]}], {acceleration}, {speed}, {radius})\n"
        self.ur_command(script_command)
        time.sleep(0.5)
        while True:
            state = self.ur_feedback.is_robot_running()
            if state is False:
                break
            time.sleep(0.1)
            
    # Funktion um den Robi mit joint angles zu steuern
    def command_angles(self, angles, acceleration=1.5, speed=0.25, radius=0):
        angles = np.array(angles)
        script_command = f"movej([{angles[0]}, {angles[1]}, {angles[2]}, {angles[0]}, {angles[1]}, {angles[2]}], {acceleration}, {speed}, {radius})\n"
        self.ur_command(script_command)
        time.sleep(0.5)
        while True:
            state = self.ur_feedback.is_robot_running()
            if state is False:
                break
            time.sleep(0.1)


    def command_gripper(self, position, speed=255, force=255):
        if 0 <= position <= 255 and 0 <= speed <= 255 and 0 <= force <= 255:
            command_pos = f'SET POS {position}\n'
            self.gripper_command(command_pos)
            command_speed = f'SET SPE {speed}\n'
            self.gripper_command(command_speed)
            command_force = f'SET FOR {force}\n'
            self.gripper_command(command_force)
            command_gto = 'SET GTO 1\n'
            self.gripper_command(command_gto)
            
            # speed = 1 -> 4sec, speed = 255 -> 0.75sec
            # Calculate base time to sleep based on speed
            time_for_speed = 4 - (3.25 * (speed - 1) / 254)
            
            # Adjust time based on the gripper position (0 to 255)
            # Assuming that the position also linearly affects the time
            time_for_position = (position / 255) * time_for_speed
            
            # Calculate the final time to sleep
            time_to_sleep = time_for_speed - time_for_position
            
            time.sleep(time_to_sleep)
        else:
            print('Invalid gripper command. Position, speed and force must be between 0 and 255.')

    def close_connections(self):
        self.socket_ur.close()
        self.socket_gripper.close()
        self.ur_feedback.disconnect()

def main():
    robot_ip = '192.168.1.10' # links ist .10, rechts .11
    # robot_ip = '192.168.56.101' ##URsim
    robot_command_port = 30002
    robot_feedback_port = 30001
    gripper_port = 63352

    ur_node = URCommand(robot_ip, robot_command_port, robot_feedback_port, gripper_port)
    try:
        while True:
            ur_node.command_position([511, -272, 175], [1.956, -3.051, 1.069], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_position([0, -350, 220], [0, -3.14, 0], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_position([511, -272, 175], [1.956, -3.051, 1.069], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_position([0, -350, 220], [0, -3.14, 0], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_position([511, -272, 175], [1.956, -3.051, 1.069], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_position([0, -350, 220], [0, -3.14, 0], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_gripper(175, speed=128, force=10)  # Example command to set gripper position
            ur_node.command_position([511, -272, 275], [1.956, -3.051, 1.069], acceleration=2.0, speed=1.5, radius=0)
            ur_node.command_gripper(0, speed=255, force=1)  # Example command to set gripper position
            ur_node.command_gripper(110, speed=255, force=10)  # Example command to set gripper position
            ur_node.command_position([0, -350, 220], [0, -3.14, 0], acceleration=2.0, speed=1.5, radius=0)
        
            # ur_node.command_angles([360, 0, 0, 0, 0, 0], acceleration=2.0, speed=1.5, radius=0) | ka was genau der macht tbh
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        ur_node.close_connections()

if __name__ == '__main__':
    main()
