import math

import rospy
import pybullet as pb
import numpy as np
from pyquaternion import Quaternion

from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped, Twist

class VRROS(object):

    def __init__(self):
        rospy.init_node('vr_callback_listener', anonymous=True)

        rospy.Subscriber('/vr/controller_3', PoseStamped, self.pose_callback)
        rospy.Subscriber('/vr/controller_buttons_3', Joy, self.button_callback)
        rospy.Subscriber('/vr/controller_velocity_3', Twist, self.velocity_callback)

        self._connected = [False, False, False]
        self._initial_position = None
        self._initial_orientation = None

        self._position = None
        self._orientation = None

        self._linear = None
        self._angular = None
        
        self._trigger = Toggler()
        self._grip = Toggler()
        self._thumbstick = Toggler(toggle_msg=['orientation locked', 'orientation unlocked'], start_state=True)
        self._buttonX = Toggler()
        self._buttonY = Toggler()

        # Quit success
        self._fail_toggle = Toggler(240, depressed_val=True)

        # Quit fail
        self._success_toggle = Toggler(240, depressed_val=True)

        self._successful = False

        self._system = 0.0

        self._want_to_quit = False

    def pose_callback(self, data):
        position = [data.pose.position.x,
                    data.pose.position.y,
                    data.pose.position.z]
        orientation = [data.pose.orientation.w,
                       data.pose.orientation.x,
                       data.pose.orientation.y,
                       data.pose.orientation.z]
        orientation = Quaternion(*orientation)

        # Scale down the position a bit 
        position = [p * 0.5 for p in position]

        # Transform between coordinate systems
        position = [position[2], position[0], position[1]]
        tf_quat = Quaternion(0.5, 0.5, 0.5, 0.5)
        orientation = tf_quat * orientation

        # Only runs during setup
        if (not self._connected[0]) and (self._initial_position is not None):
            self._connected[0] = True

        # Position is relative to initial position
        if self._initial_position is not None:
            self._position = np.asarray([p - ip for p, ip in zip(position, self._initial_position)])

            # Toggling the thumbstick will disable changing the orientation
            if self._thumbstick.toggled:
                self._orientation = self._initial_orientation
            else:
                self._initial_orientation = orientation
                self._orientation = orientation
        else: 
            self._initial_position = position
            self._initial_orientation = orientation
        
    def button_callback(self, data):
        
        # System button is wonky
        self._system = data.axes[0]

        self._trigger(data.axes[1])
        self._grip(data.axes[2])

        self._buttonX(data.buttons[0])
        self._buttonY(data.buttons[1])
        self._thumbstick(data.buttons[2])

        self._success_toggle(self.buttonY and not self.trigger)
        self._fail_toggle(self.buttonY and self.trigger)

        if (not self._connected[1]):
            self._connected[1] = True

    def velocity_callback(self, data):

        # Note the switch here
        vel = np.asarray([data.linear.z,
                          data.linear.x,
                          data.linear.y,
                          data.angular.z,
                          data.angular.x,
                          data.angular.y])

        self._linear, self._angular = vel[3:], vel[3:]

        if (not self._connected[2]):
            self._connected[2] = True

    @property
    def connected(self):
        ''' return if the vr system is connected '''
        return all(self._connected)
    
    @property
    def done(self):
        ''' indicate done by holding down the y button '''
        if (self._success_toggle.toggled):
            self._successful = True
            return True
        elif (self._fail_toggle.toggled):
            self._successful = False
            return True
        return False
    
    @property
    def successful(self):
        return self._successful

    @property
    def position(self):
        ''' return controller position '''
        return self._position

    @property
    def orientation(self):
        ''' return controller orientation '''
        return self._orientation

    @property
    def orientation_as_euler(self):
        ''' return controller orientation as euler angles '''
        return [math.degrees(p) for p in pb.getEulerFromQuaternion(self._orientation)]

    @property
    def pose(self):
        ''' return controller position and orientation '''
        return self._position, self._orientation

    @property
    def linear_velocity(self):
        ''' return controller linear velocity '''
        return self._linear

    @property
    def angular_velocity(self):
        ''' return controller angular velocity '''
        return self._angular
    
    @property
    def velocity(self):
        ''' return controller linear and angular velocity '''
        return self._linear, self._angular

    @property
    def trigger(self):
        ''' return if controller trigger is depressed '''
        return self._trigger.depressed
    
    @property
    def grip(self):
        ''' return if controller grip is depressed '''
        return self._grip.depressed
    
    @property
    def thumbstick(self):
        ''' return if controller thumbstick is depressed '''
        return self._thumbstick.depressed

    @property
    def buttonX(self):
        ''' return if controller x button is depressed '''
        return self._buttonX.depressed

    @property
    def buttonY(self):
        ''' return if controller y button is depressed '''
        return self._buttonY.depressed

class Toggler():
    def __init__(self, press_duration=30, depressed_val=3, toggle_msg=None, start_state=False):
        self._press_duration = press_duration
        self._toggle_msg = toggle_msg

        self._presses = 0
        self._value = 0
        self._depressed_val = depressed_val
        self._toggled = start_state
        self._depressed = False
        self._already_toggled = False

    def __call__(self, value):
        # Update value with new value
        self._value = value

        # Check if the button is depressed
        if self.depressed:
            if not self._already_toggled:
                self._presses += 1
                
                # Button has now been toggled
                if self._presses >= self._press_duration:
                    self._presses = 0
                    self._toggled = not self._toggled
                    self._already_toggled = True

                    # Print a toggle message
                    if self._toggle_msg is not None:
                        if self._toggled:
                            print(self._toggle_msg[0])
                        else: 
                            print(self._toggle_msg[1])
        else:
            self._presses = 0
            self._already_toggled = False

    @property
    def toggled(self):
        return self._toggled
    
    @property
    def depressed(self):
        return self._value == self._depressed_val
    
    @property
    def value(self):
        return self._value