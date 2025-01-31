from numpy import append
import rospy
import math

from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistWithCovarianceStamped, Twist, Pose
from nav_msgs.msg import Odometry
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api import geometry_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import Quaternion, SE2VelocityLimit
from bosdyn.client import math_helpers
import actionlib
import functools
import tf2_ros

from spot_msgs.msg import Metrics
from spot_msgs.msg import LeaseArray, LeaseResource
from spot_msgs.msg import FootStateArray
from spot_msgs.msg import EStopStateArray
from spot_msgs.msg import WiFiState
from spot_msgs.msg import ManipulatorState
from spot_msgs.msg import PowerState
from spot_msgs.msg import BehaviorFaultState
from spot_msgs.msg import SystemFaultState
from spot_msgs.msg import BatteryStateArray
from spot_msgs.msg import Feedback
from spot_msgs.msg import MobilityParams
from spot_msgs.msg import NavigateToAction, NavigateToResult, NavigateToFeedback
from spot_msgs.msg import TrajectoryAction, TrajectoryResult, TrajectoryFeedback
from spot_msgs.srv import ListGraph, ListGraphResponse
from spot_msgs.srv import SetLocomotion, SetLocomotionResponse
from spot_msgs.srv import ClearBehaviorFault, ClearBehaviorFaultResponse
from spot_msgs.srv import SetVelocity, SetVelocityResponse
from spot_msgs.srv import SpotPose, SpotPoseRequest, SpotPoseResponse
from spot_msgs.srv import Dock, DockResponse, GetDockState, GetDockStateResponse

from .ros_helpers import *
from .spot_wrapper import SpotWrapper

import actionlib
import logging
import threading

class SpotROS():
    """Parent class for using the wrapper.  Defines all callbacks and keeps the wrapper alive"""

    def __init__(self):
        self.callbacks = {}
        """Dictionary listing what callback to use for what data task"""
        self.callbacks["robot_state"] = self.RobotStateCB
        self.callbacks["metrics"] = self.MetricsCB
        self.callbacks["lease"] = self.LeaseCB
        self.callbacks["front_image"] = self.FrontImageCB
        self.callbacks["side_image"] = self.SideImageCB
        self.callbacks["rear_image"] = self.RearImageCB
        self.callbacks["gripper_image"] = self.GripperImageCB
        self.callbacks["world_object"] = self.WorldObjectCB

    def RobotStateCB(self, results):
        """Callback for when the Spot Wrapper gets new robot state data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        state = self.spot_wrapper.robot_state

        if state:
            ## joint states ##
            joint_state = GetJointStatesFromState(state, self.spot_wrapper)
            self.joint_state_pub.publish(joint_state)

            ## TF ##
            tf_msg = GetTFFromState(state, self.spot_wrapper, self.mode_parent_odom_tf)
            if tf_msg.transforms is not None and len(tf_msg.transforms) > 0:
                self.tf_pub.publish(tf_msg)

            # Odom Twist #
            twist_odom_msg = GetOdomTwistFromState(state, self.spot_wrapper)
            self.odom_twist_pub.publish(twist_odom_msg)

            # Odom #
            if self.mode_parent_odom_tf == 'vision':
                odom_msg = GetOdomFromState(state, self.spot_wrapper, use_vision=True)
            else:
                odom_msg = GetOdomFromState(state, self.spot_wrapper, use_vision=False)
            self.odom_pub.publish(odom_msg)

            # Feet #
            foot_array_msg = GetFeetFromState(state, self.spot_wrapper)
            self.feet_pub.publish(foot_array_msg)

            # EStop #
            estop_array_msg = GetEStopStateFromState(state, self.spot_wrapper)
            self.estop_pub.publish(estop_array_msg)

            # WIFI #
            wifi_msg = GetWifiFromState(state, self.spot_wrapper)
            self.wifi_pub.publish(wifi_msg)

            # Battery States #
            battery_states_array_msg = GetBatteryStatesFromState(state, self.spot_wrapper)
            self.battery_pub.publish(battery_states_array_msg)

            # Manipulator State #
            manipulator_state_msg = GetManipulatorStatesFromState(state, self.spot_wrapper)
            self.manipulator_pub.publish(manipulator_state_msg)

            # Power State #
            power_state_msg = GetPowerStatesFromState(state, self.spot_wrapper)
            self.power_pub.publish(power_state_msg)

            # System Faults #
            system_fault_state_msg = GetSystemFaultsFromState(state, self.spot_wrapper)
            self.system_faults_pub.publish(system_fault_state_msg)

            # Behavior Faults #
            behavior_fault_state_msg = getBehaviorFaultsFromState(state, self.spot_wrapper)
            self.behavior_faults_pub.publish(behavior_fault_state_msg)

    def MetricsCB(self, results):
        """Callback for when the Spot Wrapper gets new metrics data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        metrics = self.spot_wrapper.metrics
        if metrics:
            metrics_msg = Metrics()
            local_time = self.spot_wrapper.robotToLocalTime(metrics.timestamp)
            metrics_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)

            for metric in metrics.metrics:
                if metric.label == "distance":
                    metrics_msg.distance = metric.float_value
                if metric.label == "gait cycles":
                    metrics_msg.gait_cycles = metric.int_value
                if metric.label == "time moving":
                    metrics_msg.time_moving = rospy.Duration(metric.duration.seconds, metric.duration.nanos)
                if metric.label == "electric power":
                    metrics_msg.electric_power = rospy.Duration(metric.duration.seconds, metric.duration.nanos)

            self.metrics_pub.publish(metrics_msg)

    def LeaseCB(self, results):
        """Callback for when the Spot Wrapper gets new lease data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        lease_array_msg = LeaseArray()
        lease_list = self.spot_wrapper.lease
        if lease_list:
            for resource in lease_list:
                new_resource = LeaseResource()
                new_resource.resource = resource.resource
                new_resource.lease.resource = resource.lease.resource
                new_resource.lease.epoch = resource.lease.epoch

                for seq in resource.lease.sequence:
                    assert type(new_resource.lease.sequence) == list
                    new_resource.lease.sequence.append(seq) # type: ignore

                new_resource.lease_owner.client_name = resource.lease_owner.client_name
                new_resource.lease_owner.user_name = resource.lease_owner.user_name

                assert type(lease_array_msg.resources) == list
                lease_array_msg.resources.append(new_resource) # type: ignore

            self.lease_pub.publish(lease_array_msg)

    def FrontImageCB(self, results):
        """Callback for when the Spot Wrapper gets new front image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        data = self.spot_wrapper.front_images
        if data:
            image_msg0, camera_info_msg0 = getImageMsg(data[0], self.spot_wrapper)
            self.frontleft_image_pub.publish(image_msg0)
            self.frontleft_image_info_pub.publish(camera_info_msg0)
            image_msg1, camera_info_msg1 = getImageMsg(data[1], self.spot_wrapper)
            self.frontright_image_pub.publish(image_msg1)
            self.frontright_image_info_pub.publish(camera_info_msg1)
            image_msg2, camera_info_msg2 = getImageMsg(data[2], self.spot_wrapper)
            self.frontleft_depth_pub.publish(image_msg2)
            self.frontleft_depth_info_pub.publish(camera_info_msg2)
            image_msg3, camera_info_msg3 = getImageMsg(data[3], self.spot_wrapper)
            self.frontright_depth_pub.publish(image_msg3)
            self.frontright_depth_info_pub.publish(camera_info_msg3)

            self.populate_camera_static_transforms(data[0])
            self.populate_camera_static_transforms(data[1])
            self.populate_camera_static_transforms(data[2])
            self.populate_camera_static_transforms(data[3])

    def SideImageCB(self, results):
        """Callback for when the Spot Wrapper gets new side image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        data = self.spot_wrapper.side_images
        if data:
            image_msg0, camera_info_msg0 = getImageMsg(data[0], self.spot_wrapper)
            self.left_image_pub.publish(image_msg0)
            self.left_image_info_pub.publish(camera_info_msg0)
            image_msg1, camera_info_msg1 = getImageMsg(data[1], self.spot_wrapper)
            self.right_image_pub.publish(image_msg1)
            self.right_image_info_pub.publish(camera_info_msg1)
            image_msg2, camera_info_msg2 = getImageMsg(data[2], self.spot_wrapper)
            self.left_depth_pub.publish(image_msg2)
            self.left_depth_info_pub.publish(camera_info_msg2)
            image_msg3, camera_info_msg3 = getImageMsg(data[3], self.spot_wrapper)
            self.right_depth_pub.publish(image_msg3)
            self.right_depth_info_pub.publish(camera_info_msg3)

            self.populate_camera_static_transforms(data[0])
            self.populate_camera_static_transforms(data[1])
            self.populate_camera_static_transforms(data[2])
            self.populate_camera_static_transforms(data[3])

    def RearImageCB(self, results):
        """Callback for when the Spot Wrapper gets new rear image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        data = self.spot_wrapper.rear_images
        if data:
            image_msg0, camera_info_msg0 = getImageMsg(data[0], self.spot_wrapper)
            self.back_image_pub.publish(image_msg0)
            self.back_image_info_pub.publish(camera_info_msg0)
            mage_msg1, camera_info_msg1 = getImageMsg(data[1], self.spot_wrapper)
            self.back_depth_pub.publish(mage_msg1)
            self.back_depth_info_pub.publish(camera_info_msg1)

            self.populate_camera_static_transforms(data[0])
            self.populate_camera_static_transforms(data[1])

    def GripperImageCB(self, results):
        """Callback for when the Spot Wrapper gets new gripper image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        data = self.spot_wrapper.gripper_images
        if self.spot_wrapper.check_has_arm() and not data:
            rospy.loginfo_throttle(1, "We have an arm but no gripper image data received... if this happens at the start it's OK, otherwise it's a problem")
        if data:
            for t_data, t_image_pub, t_info_pub in zip(data, self.gripper_image_pubs, self.gripper_camera_info_pubs):
                image_msg0, camera_info_msg0 = getImageMsg(t_data, self.spot_wrapper)
                t_image_pub.publish(image_msg0)
                t_info_pub.publish(camera_info_msg0)
                self.populate_camera_static_transforms(t_data)

    def WorldObjectCB(self, results):
        """Callback for when the Spot Wrapper gets new world_object data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        del results
        proto = self.spot_wrapper.world_object
        self.publish_world_object(proto)

    def handle_claim(self, req):
        """ROS service handler for the claim service"""
        del req
        resp = self.spot_wrapper.claim()
        return TriggerResponse(resp[0], resp[1])

    def handle_force_claim(self, req):
        """ROS service handler for the forceful variant of claim service"""
        del req
        resp = self.spot_wrapper.force_claim()
        return TriggerResponse(resp[0], resp[1])

    def handle_release(self, req):
        """ROS service handler for the release service"""
        del req
        resp = self.spot_wrapper.release()
        return TriggerResponse(resp[0], resp[1])

    def handle_stop(self, req):
        """ROS service handler for the stop service"""
        del req
        resp = self.spot_wrapper.stop()
        return TriggerResponse(resp[0], resp[1])

    def handle_self_right(self, req):
        """ROS service handler for the self-right service"""
        del req
        resp = self.spot_wrapper.self_right()
        return TriggerResponse(resp[0], resp[1])

    def handle_sit(self, req):
        """ROS service handler for the sit service"""
        del req
        resp = self.spot_wrapper.sit()
        return TriggerResponse(resp[0], resp[1])

    def handle_stand(self, req):
        """ROS service handler for the stand service"""
        del req
        resp = self.spot_wrapper.stand()
        return TriggerResponse(resp[0], resp[1])

    def handle_power_on(self, req):
        """ROS service handler for the power-on service"""
        del req
        resp = self.spot_wrapper.power_on()
        return TriggerResponse(resp[0], resp[1])

    def handle_safe_power_off(self, req):
        """ROS service handler for the safe-power-off service"""
        del req
        resp = self.spot_wrapper.safe_power_off()
        return TriggerResponse(resp[0], resp[1])

    def handle_estop_hard(self, req):
        """ROS service handler to hard-eStop the robot.  The robot will immediately cut power to the motors"""
        del req
        resp = self.spot_wrapper.assertEStop(True)
        return TriggerResponse(resp[0], resp[1])

    def handle_estop_soft(self, req):
        """ROS service handler to soft-eStop the robot.  The robot will try to settle on the ground before cutting
        power to the motors """
        del req
        resp = self.spot_wrapper.assertEStop(False)
        return TriggerResponse(resp[0], resp[1])

    def handle_estop_disengage(self, req):
        """ROS service handler to disengage the eStop on the robot."""
        del req
        resp = self.spot_wrapper.disengageEStop()
        return TriggerResponse(resp[0], resp[1])

    def handle_clear_behavior_fault(self, req):
        """ROS service handler for clearing behavior faults"""
        resp = self.spot_wrapper.clear_behavior_fault(req.id)
        return ClearBehaviorFaultResponse(resp[0], resp[1])

    def handle_spot_pose(self, req:SpotPoseRequest):
        pose = {
            "euler_x" : 0.0,
            "euler_y" : 0.0,
            "euler_z" : 0.0
        }
        if req.pose_type == req.CUSTOM:
            pose['euler_x'] = req.euler_x
            pose['euler_y'] = req.euler_y
            pose['euler_z'] = req.euler_z
        if req.pose_type == req.LOOK_UP:
            pose['euler_x'] = 0
            pose['euler_y'] = -1 * math.pi / 6.0
            pose['euler_z'] = 0
        if req.pose_type == req.LOOK_DOWN:
            pose['euler_x'] = 0
            pose['euler_y'] = math.pi / 6.0
            pose['euler_z'] = 0
        if req.pose_type == req.HEAD_TILT:
            pose['euler_x'] = math.pi / 6.0
            pose['euler_y'] = 0
            pose['euler_z'] = 0
        if req.pose_type == req.SIT:
            resp = self.handle_sit(None)
            return SpotPoseResponse(resp.success, resp.message)
        if req.pose_type == req.STAND:
            resp = self.handle_stand(None)
            return SpotPoseResponse(resp.success, resp.message)

        timeout = 0
        if req.pose_hold_time >= 0:
            timeout = req.pose_hold_time
        else:
            if req.pose_hold_time == req.POSE_HOLD_TIME_INF:
                timeout = -1  # anything negative will mean hold
            if req.pose_hold_time == req.POSE_HOLD_TIME_DEFAULT:
                timeout = 10  # TODO: parmeterize --cst

        # TODO: parameterize, support height param, timeout, other poses --cst
        try:
            rospy.loginfo("Told to achieve pose {} for {}".format(pose, timeout))
            self.spot_wrapper.spot_pose(pose['euler_x'], pose['euler_y'], pose['euler_z'], timeout_sec=timeout)
            return SpotPoseResponse(True, 'Success')
        except Exception as e:
            return SpotPoseResponse(False, 'Error:{}'.format(e))

    def handle_stair_mode(self, req):
        """ROS service handler to set a stair mode to the robot."""
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.stair_hint = req.data
            self.spot_wrapper.set_mobility_params( mobility_params )
            return SetBoolResponse(True, 'Success')
        except Exception as e:
            return SetBoolResponse(False, 'Error:{}'.format(e))

    def handle_locomotion_mode(self, req):
        """ROS service handler to set locomotion mode"""
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.locomotion_hint = req.locomotion_mode
            self.spot_wrapper.set_mobility_params( mobility_params )
            return SetLocomotionResponse(True, 'Success')
        except Exception as e:
            return SetLocomotionResponse(False, 'Error:{}'.format(e))

    def handle_max_vel(self, req):
        """
        Handle a max_velocity service call. This will modify the mobility params to have a limit on the maximum
        velocity that the robot can move during motion commmands. This affects trajectory commands and velocity
        commands

        Args:
            req: SetVelocityRequest containing requested maximum velocity

        Returns: SetVelocityResponse
        """
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.vel_limit.CopyFrom(SE2VelocityLimit(max_vel=math_helpers.SE2Velocity(req.velocity_limit.linear.x,
                                                                                                 req.velocity_limit.linear.y,
                                                                                                 req.velocity_limit.angular.z).to_proto()))
            self.spot_wrapper.set_mobility_params(mobility_params)
            return SetVelocityResponse(True, 'Success')
        except Exception as e:
            return SetVelocityResponse(False, 'Error:{}'.format(e))

    def handle_trajectory(self, req):
        """ROS actionserver execution handler to handle receiving a request to move to a location"""
        if req.target_pose.header.frame_id != 'body':
            self.trajectory_server.set_aborted(TrajectoryResult(False, 'frame_id of target_pose must be \'body\''))
            return
        if req.duration.data.to_sec() <= 0:
            self.trajectory_server.set_aborted(TrajectoryResult(False, 'duration must be larger than 0'))
            return

        cmd_duration = rospy.Duration(req.duration.data.secs, req.duration.data.nsecs)
        resp = self.spot_wrapper.trajectory_cmd(
                        goal_x=req.target_pose.pose.position.x,
                        goal_y=req.target_pose.pose.position.y,
                        goal_heading=math_helpers.Quat(
                            w=req.target_pose.pose.orientation.w,
                            x=req.target_pose.pose.orientation.x,
                            y=req.target_pose.pose.orientation.y,
                            z=req.target_pose.pose.orientation.z
                            ).to_yaw(),
                        cmd_duration=cmd_duration.to_sec(),
                        precise_position=req.precise_positioning,
                        )

        def timeout_cb(trajectory_server, _):
            trajectory_server.publish_feedback(TrajectoryFeedback("Failed to reach goal, timed out"))
            trajectory_server.set_aborted(TrajectoryResult(False, "Failed to reach goal, timed out"))

        # Abort the actionserver if cmd_duration is exceeded - the driver stops but does not provide feedback to
        # indicate this so we monitor it ourselves
        cmd_timeout = rospy.Timer(cmd_duration, functools.partial(timeout_cb, self.trajectory_server), oneshot=True)

        # The trajectory command is non-blocking but we need to keep this function up in order to interrupt if a
        # preempt is requested and to return success if/when the robot reaches the goal. Also check the is_active to
        # monitor whether the timeout_cb has already aborted the command
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.trajectory_server.is_preempt_requested() and not self.spot_wrapper.at_goal and self.trajectory_server.is_active():
            if self.spot_wrapper.near_goal:
                if self.spot_wrapper._last_trajectory_command_precise:
                    self.trajectory_server.publish_feedback(TrajectoryFeedback("Near goal, performing final adjustments"))
                else:
                    self.trajectory_server.publish_feedback(TrajectoryFeedback("Near goal"))
            else:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Moving to goal"))
            rate.sleep()

        # If still active after exiting the loop, the command did not time out
        if self.trajectory_server.is_active():
            cmd_timeout.shutdown()
            if self.trajectory_server.is_preempt_requested():
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Preempted"))
                self.trajectory_server.set_preempted()
                self.spot_wrapper.stop()

            if self.spot_wrapper.at_goal:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Reached goal"))
                self.trajectory_server.set_succeeded(TrajectoryResult(resp[0], resp[1]))
            else:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Failed to reach goal"))
                self.trajectory_server.set_aborted(TrajectoryResult(False, "Failed to reach goal"))

    def handle_dock(self, req):
        """Dock the robot"""
        resp = self.spot_wrapper.dock(req.dock_id)
        return DockResponse(resp[0], resp[1])

    def handle_undock(self, req):
        """Undock the robot"""
        resp = self.spot_wrapper.undock()
        return TriggerResponse(resp[0], resp[1])

    def handle_get_docking_state(self, req):
        """Get docking state of robot"""
        resp = self.spot_wrapper.get_docking_state()
        return GetDockStateResponse(GetDockStatesFromState(resp))

    def handle_roll_over_right(self, req):
        """Robot sit down and roll on to it its side for easier battery access"""
        del req
        resp = self.spot_wrapper.battery_change_pose(1)
        return TriggerResponse(resp[0], resp[1])

    def handle_roll_over_left(self, req):
        """Robot sit down and roll on to it its side for easier battery access"""
        del req
        resp = self.spot_wrapper.battery_change_pose(2)
        return TriggerResponse(resp[0], resp[1])

    def cmdVelCallback(self, data):
        """Callback for cmd_vel command"""
        self.spot_wrapper.velocity_cmd(data.linear.x, data.linear.y, data.angular.z)

    def bodyPoseCallback(self, data):
        """Callback for cmd_vel command"""
        q = Quaternion()
        q.x = data.orientation.x
        q.y = data.orientation.y
        q.z = data.orientation.z
        q.w = data.orientation.w
        position = geometry_pb2.Vec3(z=data.position.z)
        pose = geometry_pb2.SE3Pose(position=position, rotation=q)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        body_control = spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

        mobility_params = self.spot_wrapper.get_mobility_params()
        mobility_params.body_control.CopyFrom(body_control)
        self.spot_wrapper.set_mobility_params(mobility_params)

    def handle_list_graph(self, upload_path):
        """ROS service handler for listing graph_nav waypoint_ids"""
        resp = self.spot_wrapper.list_graph(upload_path)
        return ListGraphResponse(resp)

    def handle_navigate_to_feedback(self):
        """Thread function to send navigate_to feedback"""
        while not rospy.is_shutdown() and self.run_navigate_to:
            localization_state = self.spot_wrapper._graph_nav_client.get_localization_state()
            if localization_state.localization.waypoint_id:
                self.navigate_as.publish_feedback(NavigateToFeedback(localization_state.localization.waypoint_id))
            rospy.Rate(10).sleep()

    def handle_navigate_to(self, msg):
        """ROS service handler to run mission of the robot.  The robot will replay a mission"""
        # create thread to periodically publish feedback
        feedback_thraed = threading.Thread(target = self.handle_navigate_to_feedback, args = ())
        self.run_navigate_to = True
        feedback_thraed.start()
        # run navigate_to
        resp = self.spot_wrapper.navigate_to(upload_path = msg.upload_path,
                                             navigate_to = msg.navigate_to,
                                             initial_localization_fiducial = msg.initial_localization_fiducial,
                                             initial_localization_waypoint = msg.initial_localization_waypoint)
        self.run_navigate_to = False
        feedback_thraed.join()

        # check status
        if resp[0]:
            self.navigate_as.set_succeeded(NavigateToResult(resp[0], resp[1]))
        else:
            self.navigate_as.set_aborted(NavigateToResult(resp[0], resp[1]))

    def populate_camera_static_transforms(self, image_data):
        """Check data received from one of the image tasks and use the transform snapshot to extract the camera frame
        transforms. This is the transforms from body->frontleft->frontleft_fisheye, for example. These transforms
        never change, but they may be calibrated slightly differently for each robot so we need to generate the
        transforms at runtime.

        Args:
        image_data: Image protobuf data from the wrapper
        """
        # We exclude the odometry frames from static transforms since they are not static. We can ignore the body
        # frame because it is a child of odom or vision depending on the mode_parent_odom_tf, and will be published
        # by the non-static transform publishing that is done by the state callback
        excluded_frames = [self.tf_name_vision_odom, self.tf_name_kinematic_odom, "body"]
        for frame_name in image_data.shot.transforms_snapshot.child_to_parent_edge_map:
            if frame_name in excluded_frames:
                continue
            parent_frame = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name
            existing_transforms = [(transform.header.frame_id, transform.child_frame_id) for transform in self.camera_static_transforms]
            if (parent_frame, frame_name) in existing_transforms:
                # We already extracted this transform
                continue

            transform = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
            local_time = self.spot_wrapper.robotToLocalTime(image_data.shot.acquisition_time)
            tf_time = rospy.Time(local_time.seconds, local_time.nanos)
            static_tf = populateTransformStamped(tf_time, transform.parent_frame_name, frame_name,
                                                 transform.parent_tform_child)
            self.camera_static_transforms.append(static_tf)
            self.camera_static_transform_broadcaster.sendTransform(self.camera_static_transforms)

    def publish_world_object(self, proto):
        bbox = BoundingBoxArray()
        bbox.header.frame_id = "vision"
        bbox.header.stamp = rospy.Time.now()
        bbox_detection = BoundingBoxArray()
        bbox_detection.header.frame_id = "vision"
        bbox_detection.header.stamp = rospy.Time.now()
        for world_object in proto.world_objects:
            rospy.logdebug(f"world_object {world_object}")
            if world_object.name == "world_obj_tracked_entity":
                timestamp = rospy.Time(
                    secs=world_object.acquisition_time.seconds,
                    nsecs=world_object.acquisition_time.nanos,
                )
                for key in world_object.transforms_snapshot.child_to_parent_edge_map:
                    frame = world_object.transforms_snapshot.child_to_parent_edge_map[
                        key
                    ]
                    if key.startswith("blob"):
                        if not key.endswith("detection"):
                            box = BoundingBox()
                            box.header.frame_id = frame.parent_frame_name
                            box.header.stamp = timestamp
                            box.label = (
                                abs(hash(key)) % 2**32
                            )  # Round hashed value to the range of uint32
                            box.pose.position.x = frame.parent_tform_child.position.x
                            box.pose.position.y = frame.parent_tform_child.position.y
                            box.pose.position.z = frame.parent_tform_child.position.z - 0.8
                            box.pose.orientation.w = frame.parent_tform_child.rotation.w
                            box.pose.orientation.x = frame.parent_tform_child.rotation.x
                            box.pose.orientation.y = frame.parent_tform_child.rotation.y
                            box.pose.orientation.z = frame.parent_tform_child.rotation.z
                            box.dimensions.x = 1.0
                            box.dimensions.y = 1.0
                            box.dimensions.z = 0.1
                            bbox.boxes.append(box)
                        else:
                            box = BoundingBox()
                            box.header.frame_id = frame.parent_frame_name
                            box.header.stamp = timestamp
                            box.label = (
                                abs(hash(key)) % 2**32
                            )  # Round hashed value to the range of uint32
                            box.pose.position.x = frame.parent_tform_child.position.x
                            box.pose.position.y = frame.parent_tform_child.position.y
                            box.pose.position.z = frame.parent_tform_child.position.z - 0.8
                            box.pose.orientation.w = frame.parent_tform_child.rotation.w
                            box.pose.orientation.x = frame.parent_tform_child.rotation.x
                            box.pose.orientation.y = frame.parent_tform_child.rotation.y
                            box.pose.orientation.z = frame.parent_tform_child.rotation.z
                            box.dimensions.x = 1.0
                            box.dimensions.y = 1.0
                            box.dimensions.z = 0.1
                            bbox_detection.boxes.append(box)
        self.world_object_bbox_pub.publish(bbox)
        self.world_object_detection_bbox_pub.publish(bbox_detection)

    def shutdown(self):
        rospy.loginfo("Shutting down ROS driver for Spot")
        if self.spot_wrapper.check_has_arm():
            self.spot_wrapper._arm.handle_stow_arm(None)
        self.spot_wrapper.sit()
        rospy.Rate(0.25).sleep()
        self.spot_wrapper.disconnect()

    def main(self):
        """Main function for the SpotROS class.  Gets config from ROS and initializes the wrapper.  Holds lease from wrapper and updates all async tasks at the ROS rate"""
        rospy.init_node('spot_ros', anonymous=True)
        rate = rospy.Rate(50)

        self.rates = rospy.get_param('~rates', {})
        self.username = rospy.get_param('~username', 'default_value')
        self.password = rospy.get_param('~password', 'default_value')
        self.hostname = rospy.get_param('~hostname', 'default_value')
        self.motion_deadzone = rospy.get_param('~deadzone', 0.05)
        self.estop_timeout = rospy.get_param('~estop_timeout', 9.0)

        self.camera_static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # Static transform broadcaster is super simple and just a latched publisher. Every time we add a new static
        # transform we must republish all static transforms from this source, otherwise the tree will be incomplete.
        # We keep a list of all the static transforms we already have so they can be republished, and so we can check
        # which ones we already have
        self.camera_static_transforms = []

        # Spot has 2 types of odometries: 'odom' and 'vision'
        # The former one is kinematic odometry and the second one is a combined odometry of vision and kinematics
        # These params enables to change which odometry frame is a parent of body frame and to change tf names of each odometry frames.
        self.mode_parent_odom_tf = rospy.get_param('~mode_parent_odom_tf', 'vision') # 'vision' or 'odom'
        self.tf_name_kinematic_odom = rospy.get_param('~tf_name_kinematic_odom', 'odom')
        self.tf_name_raw_kinematic = 'odom'
        self.tf_name_vision_odom = rospy.get_param('~tf_name_vision_odom', 'vision')
        self.tf_name_raw_vision = 'vision'
        if self.mode_parent_odom_tf != self.tf_name_raw_kinematic and self.mode_parent_odom_tf != self.tf_name_raw_vision:
            rospy.logerr('rosparam \'~mode_parent_odom_tf\' should be \'odom\' or \'vision\'.')
            return

        self.logger = logging.getLogger('rosout')

        rospy.loginfo("Starting ROS driver for Spot at "+str(self.hostname)+" as "+str(self.username)+" "+str(self.password))
        self.spot_wrapper = SpotWrapper(self.username, self.password, self.hostname, self.logger, self.estop_timeout, self.rates, self.callbacks)

        if not self.spot_wrapper.is_valid:
            rospy.logerr("Failed to connect to Spot")
            return

        if self.spot_wrapper.check_has_arm():
            rospy.loginfo("Arm detected, started that too")

        if self.spot_wrapper.is_valid:
            # Images #
            self.back_image_pub = rospy.Publisher('camera/back/image', Image, queue_size=10)
            self.frontleft_image_pub = rospy.Publisher('camera/frontleft/image', Image, queue_size=10)
            self.frontright_image_pub = rospy.Publisher('camera/frontright/image', Image, queue_size=10)
            self.left_image_pub = rospy.Publisher('camera/left/image', Image, queue_size=10)
            self.right_image_pub = rospy.Publisher('camera/right/image', Image, queue_size=10)

            # Depth #
            self.back_depth_pub = rospy.Publisher('depth/back/image', Image, queue_size=10)
            self.frontleft_depth_pub = rospy.Publisher('depth/frontleft/image', Image, queue_size=10)
            self.frontright_depth_pub = rospy.Publisher('depth/frontright/image', Image, queue_size=10)
            self.left_depth_pub = rospy.Publisher('depth/left/image', Image, queue_size=10)
            self.right_depth_pub = rospy.Publisher('depth/right/image', Image, queue_size=10)

            # Image Camera Info #
            self.back_image_info_pub = rospy.Publisher('camera/back/camera_info', CameraInfo, queue_size=10)
            self.frontleft_image_info_pub = rospy.Publisher('camera/frontleft/camera_info', CameraInfo, queue_size=10)
            self.frontright_image_info_pub = rospy.Publisher('camera/frontright/camera_info', CameraInfo, queue_size=10)
            self.left_image_info_pub = rospy.Publisher('camera/left/camera_info', CameraInfo, queue_size=10)
            self.right_image_info_pub = rospy.Publisher('camera/right/camera_info', CameraInfo, queue_size=10)
            # Depth Camera Info #
            self.back_depth_info_pub = rospy.Publisher('depth/back/camera_info', CameraInfo, queue_size=10)
            self.frontleft_depth_info_pub = rospy.Publisher('depth/frontleft/camera_info', CameraInfo, queue_size=10)
            self.frontright_depth_info_pub = rospy.Publisher('depth/frontright/camera_info', CameraInfo, queue_size=10)
            self.left_depth_info_pub = rospy.Publisher('depth/left/camera_info', CameraInfo, queue_size=10)
            self.right_depth_info_pub = rospy.Publisher('depth/right/camera_info', CameraInfo, queue_size=10)

            self.gripper_image_pubs = []
            self.gripper_camera_info_pubs = []
            if self.spot_wrapper.check_has_arm():
                for t in ['hand_color', 'hand_depth', 'hand_image', 'hand_depth_in_hand_color_frame', 'hand_color_in_hand_depth_frame']:
                    self.gripper_image_pubs.append(rospy.Publisher('camera/'+t+'/image', Image, queue_size=10))
                    self.gripper_camera_info_pubs.append(rospy.Publisher('camera/'+t+'/camera_info', CameraInfo, queue_size=10))

            # World Object #
            # world_object_bbox_pub publishes the pose and bounding box of recognized_objects which are obtained from bosdyn's world_object api
            self.world_object_bbox_pub = rospy.Publisher(
                "tracked_world_objects", BoundingBoxArray, queue_size=10
            )
            self.world_object_detection_bbox_pub = rospy.Publisher(
                "tracked_world_objects_detection", BoundingBoxArray, queue_size=10
            )

            # Status Publishers #
            self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
            """Defining a TF publisher manually because of conflicts between Python3 and tf"""
            self.tf_pub = rospy.Publisher('tf', TFMessage, queue_size=10)
            self.metrics_pub = rospy.Publisher('status/metrics', Metrics, queue_size=10)
            self.lease_pub = rospy.Publisher('status/leases', LeaseArray, queue_size=10)
            self.odom_twist_pub = rospy.Publisher('odometry/twist', TwistWithCovarianceStamped, queue_size=10)
            self.odom_pub = rospy.Publisher('odometry', Odometry, queue_size=10)
            self.feet_pub = rospy.Publisher('status/feet', FootStateArray, queue_size=10)
            self.estop_pub = rospy.Publisher('status/estop', EStopStateArray, queue_size=10)
            self.wifi_pub = rospy.Publisher('status/wifi', WiFiState, queue_size=10)
            self.manipulator_pub = rospy.Publisher('status/manipulator_state', ManipulatorState, queue_size=10)
            self.power_pub = rospy.Publisher('status/power_state', PowerState, queue_size=10)
            self.battery_pub = rospy.Publisher('status/battery_states', BatteryStateArray, queue_size=10)
            self.behavior_faults_pub = rospy.Publisher('status/behavior_faults', BehaviorFaultState, queue_size=10)
            self.system_faults_pub = rospy.Publisher('status/system_faults', SystemFaultState, queue_size=10)

            self.feedback_pub = rospy.Publisher('status/feedback', Feedback, queue_size=10)

            self.mobility_params_pub = rospy.Publisher('status/mobility_params', MobilityParams, queue_size=10)

            rospy.Subscriber('cmd_vel', Twist, self.cmdVelCallback, queue_size = 1)
            rospy.Subscriber('body_pose', Pose, self.bodyPoseCallback, queue_size = 1)

            rospy.Service("claim", Trigger, self.handle_claim)
            rospy.Service("force_claim", Trigger, self.handle_force_claim)
            rospy.Service("release", Trigger, self.handle_release)
            rospy.Service("stop", Trigger, self.handle_stop)
            rospy.Service("self_right", Trigger, self.handle_self_right)
            rospy.Service("sit", Trigger, self.handle_sit)
            rospy.Service("stand", Trigger, self.handle_stand)
            rospy.Service("power_on", Trigger, self.handle_power_on)
            rospy.Service("power_off", Trigger, self.handle_safe_power_off)

            rospy.Service("estop/hard", Trigger, self.handle_estop_hard)
            rospy.Service("estop/gentle", Trigger, self.handle_estop_soft)
            rospy.Service("estop/release", Trigger, self.handle_estop_disengage)

            rospy.Service("spot_pose", SpotPose, self.handle_spot_pose)
            rospy.Service("stair_mode", SetBool, self.handle_stair_mode)
            rospy.Service("locomotion_mode", SetLocomotion, self.handle_locomotion_mode)
            rospy.Service("max_velocity", SetVelocity, self.handle_max_vel)
            rospy.Service("clear_behavior_fault", ClearBehaviorFault, self.handle_clear_behavior_fault)

            rospy.Service("list_graph", ListGraph, self.handle_list_graph)

            # Docking
            rospy.Service("dock", Dock, self.handle_dock)
            rospy.Service("undock", Trigger, self.handle_undock)
            rospy.Service("docking_state", GetDockState, self.handle_get_docking_state)

            # Roll Over
            rospy.Service("roll_over_right", Trigger, self.handle_roll_over_right)
            rospy.Service("roll_over_left", Trigger, self.handle_roll_over_left)

            self.navigate_as = actionlib.SimpleActionServer('navigate_to', NavigateToAction,
                                                            execute_cb = self.handle_navigate_to,
                                                            auto_start = False)
            self.navigate_as.start()

            self.trajectory_server = actionlib.SimpleActionServer("trajectory", TrajectoryAction,
                                                                  execute_cb=self.handle_trajectory,
                                                                  auto_start=False)
            self.trajectory_server.start()

            rospy.on_shutdown(self.shutdown)

            self.auto_claim = rospy.get_param('~auto_claim', False)
            self.auto_power_on = rospy.get_param('~auto_power_on', False)
            self.auto_stand = rospy.get_param('~auto_stand', False)

            if self.auto_claim:
                self.spot_wrapper.claim()
                if self.auto_power_on:
                    self.spot_wrapper.power_on()
                    if self.auto_stand:
                        self.spot_wrapper.stand()

            while not rospy.is_shutdown():
                self.spot_wrapper.updateTasks()
                feedback_msg = Feedback()
                feedback_msg.standing = self.spot_wrapper.is_standing
                feedback_msg.sitting = self.spot_wrapper.is_sitting
                feedback_msg.moving = self.spot_wrapper.is_moving
                id = self.spot_wrapper.id
                try:
                    feedback_msg.serial_number = id.serial_number
                    feedback_msg.species = id.species
                    feedback_msg.version = id.version
                    feedback_msg.nickname = id.nickname
                    feedback_msg.computer_serial_number = id.computer_serial_number
                except:
                    pass
                self.feedback_pub.publish(feedback_msg)
                mobility_params_msg = MobilityParams()
                try:
                    mobility_params = self.spot_wrapper.get_mobility_params()
                    mobility_params_msg.body_control.position.x = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.position.x
                    mobility_params_msg.body_control.position.y = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.position.y
                    mobility_params_msg.body_control.position.z = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.position.z
                    mobility_params_msg.body_control.orientation.x = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.x
                    mobility_params_msg.body_control.orientation.y = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.y
                    mobility_params_msg.body_control.orientation.z = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.z
                    mobility_params_msg.body_control.orientation.w = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.w
                    mobility_params_msg.locomotion_hint = mobility_params.locomotion_hint
                    mobility_params_msg.stair_hint = mobility_params.stair_hint
                except Exception as e:
                    rospy.logerr('Error:{}'.format(e))
                    pass
                self.mobility_params_pub.publish(mobility_params_msg)
                rate.sleep()
