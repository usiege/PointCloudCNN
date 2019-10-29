#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:19-3-20 下午4:09
# software:PyCharm

from cnn import *

class MPLNet(CNN):
	"""docstring for MPLNet"""
	def __init__(self, arg):
		super(MPLNet, self).__init__()
		self.arg = arg
		
	