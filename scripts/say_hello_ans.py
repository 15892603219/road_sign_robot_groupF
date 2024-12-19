#!/usr/bin/env python

"""
	To run this script, you need to run below lines in terminal:
	-[Robot] $ roslaunch rchomeedu_vision multi_astra.launch
	-[IO] $ roslaunch astra_camera astra.launch
	-[Juno] $ roslaunch usb_cam usb_cam-test.launch
	- roslaunch opencv_apps face_detection.launch image:=/camera/rgb/image_raw
	- roslaunch kids_module say_hello.launch
"""

import rospy
from sound_play.libsoundplay import SoundClient
from opencv_apps.msg import FaceArrayStamped

greet_flag = 0
# greet_flag = 0 means haven't greet the guest
# greet_flag = 1 means greeted the guest

class SayHello:
	def __init__(self):
		rospy.init_node('say_hello')
		rospy.on_shutdown(self.cleanup)

		# Create the sound client object
		self.soundhandle = SoundClient()
		# self.soundhandle = SoundClient(blocking=True)
		
		# Wait a moment to let the client connect to the sound_play server
		rospy.sleep(1)
		
		# Make sure any lingering sound_play processes are stopped.
		self.soundhandle.stopAll()
		rospy.loginfo("Ready, waiting for commands...")

		# Subscribe to the face detection output and set the callback function
		rospy.Subscriber('/face_detection/faces', FaceArrayStamped, self.talkback)
	
	def cleanup(self):
		self.soundhandle.stopAll()
		rospy.loginfo("Shutting down partybot node...")

	def talkback(self, msg):
		faces = msg.faces
		# ======================== YOUR CODE HERE ========================
		# Instruction: Say Hello when face(s) is detected
		
		# Use global flag to let the robot will only greet people once
		global greet_flag
		
		# When the robot have not greet the guest
		if greet_flag == 0:
			# Check whether face(s) is detected
			if len(faces) > 0:
				# Since a robot may detect more than 1 face,
				# declare faces_data and eyes_data lists 
				# to access all face and eyes data
				faces_data = list()
				eyes_data = list()
				for i in faces:
					faces_data.append(i.face)
					
					# Eyes data will only listed if eyes are detected
					if i.eyes:
						rospy.loginfo(i.eyes)
						eyes_data.append(i.eyes)
					else:
						eyes_data.append("null")
				
				rospy.loginfo(eyes_data)
				# Will only greet the person if eyes are detected
				if eyes_data and [x for x in eyes_data if x != "null" ]:	
					self.soundhandle.say("Good morning. How can I help you?")
					greet_flag = 1
		
		else:
			# Reset the flag if face not detected
			if len(faces) < 1:
				greet_flag = 0
		# ================================================================

if __name__=="__main__":
	try:
		SayHello()
		rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Partybot node terminated.")
