#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <Eigen/Dense>
#include <thread> 
#include <std_msgs/msg/bool.hpp>

class HandPositionFollower : public rclcpp::Node {
public:
    HandPositionFollower()
    : Node("moveit_mover") {
        // OMPL Parameter deklarieren
        this->declare_parameter("robot_description_planning.default_planner_config", "RRTConnect");
        this->declare_parameter("robot_description_planning.default_planner_config.settings", "RRTConnectkConfigDefault");

        // Optional: Timeout und Planungszeit konfigurieren
        this->declare_parameter("robot_description_planning.ompl_planning.timeout", 10.0);
        this->declare_parameter("robot_description_planning.ompl_planning.max_planning_attempts", 10);

        // Kinematik-Plugin deklarieren
        this->declare_parameter("robot_description_kinematics.ur_manipulator.kinematics_solver", "kdl_kinematics_plugin/KDLKinematicsPlugin");
        this->declare_parameter("robot_description_kinematics.ur_manipulator.kinematics_solver_search_resolution", 0.005);
        this->declare_parameter("robot_description_kinematics.ur_manipulator.kinematics_solver_timeout", 0.05);

        // Abonnieren des /hand_position Topics
        hand_position_sub_ = this->create_subscription<geometry_msgs::msg::Pose>("/hand_pose", 10,
        std::bind(&HandPositionFollower::handPositionCallback, this, std::placeholders::_1));

        // Gripper publisher initialisieren
        gripper_pub_ = this->create_publisher<std_msgs::msg::Bool>("/gripper_command", 10);

        RCLCPP_INFO(this->get_logger(), "Moveit Mover Node initialized.");
    }

    // Initialisiere die MoveGroupInterface
    void initializeMoveGroupInterface() {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "ur_manipulator");

        // Konfiguriere MoveIt Parameter
        move_group_->setPlanningTime(5.0);
        move_group_->setMaxVelocityScalingFactor(1);
        move_group_->setMaxAccelerationScalingFactor(0.4);

        RCLCPP_INFO(this->get_logger(), "MoveGroupInterface initialized.");

        // Gripper öffnen (false)
        RCLCPP_INFO(this->get_logger(), "Opening gripper...");
        publishGripperCommand(false);
        std::this_thread::sleep_for(std::chrono::seconds(1));  // Kurze Wartezeit für den Gripper

        // Fahre zur Startposition
        moveToObjectPosition();

        // Gripper schließen (true)
        RCLCPP_INFO(this->get_logger(), "Closing gripper...");
        publishGripperCommand(true);
        std::this_thread::sleep_for(std::chrono::seconds(1));  // Kurze Wartezeit für den Gripper

        // Über das Objekt fahren
        moveAboveObject();
    }

private:
    void handPositionCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        // Zielpose definieren
        geometry_msgs::msg::Pose target_pose;

        double x_pos = msg->position.x;
        double y_pos = msg->position.y;
        double z_pos = msg->position.z;
        
        RCLCPP_INFO(this->get_logger(), "Received hand position: x=%.3f, y=%.3f, z=%.3f", x_pos, y_pos, z_pos);

        // target_pose.position = msg->position;
        target_pose.position.x = x_pos; 
        target_pose.position.y = y_pos;
        target_pose.position.z = z_pos + 0.25;
        target_pose.orientation = msg->orientation;

        // Zielpose setzen
        move_group_->setPoseTarget(target_pose);

        // Bewegung planen und ausführen
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Planning successful, executing...");
            move_group_->execute(plan);

            // Greifer öffnen (false)
            RCLCPP_INFO(this->get_logger(), "Opening gripper after reaching target pose...");
            publishGripperCommand(false);

            // Wartezeit nach Zielerreichung (2 Sekunden)
            RCLCPP_INFO(this->get_logger(), "Waiting for 4 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Rückkehr zur Home-Position über Joint-Winkel
            moveToHomePositionUsingJoints();
        } else {
            RCLCPP_ERROR(this->get_logger(), "Planning failed.");
        }
    }

    // Bewegung zur Home-Position über Joint-Winkel
    void moveToHomePositionUsingJoints() {
        // Definiere Joint-Winkel für die Home-Position
        std::vector<double> home_joint_positions = {
            0,    // Joint 1: 0°
            -M_PI_2, // Joint 2: 90°
            0.0,    // Joint 3: 0°
            0.0,    // Joint 4: 0°
            0.0,    // Joint 5: 0°
            0.0     // Joint 6: 0°
        };

        // Setze Joint-Winkel als Ziel
        move_group_->setJointValueTarget(home_joint_positions);

        // Bewegung planen und ausführen
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Planning to Home-Position (Joints) successful, executing...");
            move_group_->execute(plan);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Planning to Home-Position (Joints) failed.");
        }
    }

    void moveToObjectPosition() {
    // Definiere die Zielpose für die Startposition
        geometry_msgs::msg::Pose start_pose;
        start_pose.position.x = 0.4;
        start_pose.position.y = 0.2;
        start_pose.position.z = 0.27;
        start_pose.orientation.x = 1.0;
        start_pose.orientation.y = 0.0;
        start_pose.orientation.z = 0.0;
        start_pose.orientation.w = 0.0;

        // Zielpose setzen
        move_group_->setPoseTarget(start_pose);

        // Bewegung planen und ausführen
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Planning to Start Position successful, executing...");
            move_group_->execute(plan);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Planning to Start Position failed.");
        }
    }

    void moveAboveObject() {
        geometry_msgs::msg::Pose start_pose;
        start_pose.position.x = 0.4;
        start_pose.position.y = 0.2;
        start_pose.position.z = 0.33;
        start_pose.orientation.x = 1.0;
        start_pose.orientation.y = 0.0;
        start_pose.orientation.z = 0.0;
        start_pose.orientation.w = 0.0;

        // Zielpose setzen
        move_group_->setPoseTarget(start_pose);

        // Bewegung planen und ausführen
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Planning to Start Position successful, executing...");
            move_group_->execute(plan);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Planning to Start Position failed.");
        }
    }

    void publishGripperCommand(bool close) {
        auto msg = std_msgs::msg::Bool();
        msg.data = close;
        gripper_pub_->publish(msg);
    }

    // Subscriber für /hand_position Topic
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr hand_position_sub_;

    // MoveGroupInterface für Robotersteuerung
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

    // Publisher für Gripper-Befehle
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr gripper_pub_;
};

int main(int argc, char** argv) {
    // ROS 2 initialisieren
    rclcpp::init(argc, argv);

    // Node erstellen und MoveGroupInterface initialisieren
    auto node = std::make_shared<HandPositionFollower>();
    node->initializeMoveGroupInterface();

    // Node ausführen
    rclcpp::spin(node);

    // ROS 2 beenden
    rclcpp::shutdown();
    return 0;
}
