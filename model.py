import numpy as np
import numpy.matlib
import math
import random
import os
import sys
import xml.etree.ElementTree as ET

import tensorflow as tf
from utils import *

class Model():
	def __init__(self, args, logger):
		self.logger = logger

		# ----- transfer some of the args params over to the model

		# model params
		self.rnn_size = args.rnn_size
		self.train = args.train
		self.nmixtures = args.nmixtures
		self.kmixtures = args.kmixtures
		self.batch_size = args.batch_size if self.train else 1 # training/sampling specific
		self.tsteps = args.tsteps if self.train else 1 # training/sampling specific
		self.alphabet = args.alphabet
		self.pjoints = args.pjoints
		self.ndimensions = args.ndimensions
		self.c = self.pjoints*self.ndimensions	
		# training params
		self.dropout = args.dropout
		self.grad_clip = args.grad_clip
		# misc
		self.tsteps_per_ascii = args.tsteps_per_ascii
		self.data_dir = args.data_dir

		self.graves_initializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)



		# ----- build the basic recurrent network architecture
		cell_func = tf.contrib.rnn.LSTMCell # could be GRUCell or RNNCell
		self.cell0 = cell_func(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)
		self.cell1 = cell_func(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)
		self.cell2 = cell_func(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)

		if (self.train and self.dropout < 1): # training mode
			self.cell0 = tf.contrib.rnn.DropoutWrapper(self.cell0, output_keep_prob = self.dropout)
			self.cell1 = tf.contrib.rnn.DropoutWrapper(self.cell1, output_keep_prob = self.dropout)
			self.cell2 = tf.contrib.rnn.DropoutWrapper(self.cell2, output_keep_prob = self.dropout)

		self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, self.tsteps, self.c])
		self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, self.tsteps, self.c])
		self.istate_cell0 = self.cell0.zero_state(batch_size=self.batch_size, dtype=tf.float32)
		self.istate_cell1 = self.cell1.zero_state(batch_size=self.batch_size, dtype=tf.float32)
		self.istate_cell2 = self.cell2.zero_state(batch_size=self.batch_size, dtype=tf.float32)

		#slice the input volume into separate vols for each tstep
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(self.input_data, self.tsteps, 1)]
		#build cell0 computational graph
		outs_cell0, self.fstate_cell0 = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.istate_cell0, self.cell0, loop_function=None, scope='cell0')


	# ----- finish building LSTMs 2 and 3
		outs_cell1, self.fstate_cell1 = tf.contrib.legacy_seq2seq.rnn_decoder(outs_cell0, self.istate_cell1, self.cell1, loop_function=None, scope='cell1')

		outs_cell2, self.fstate_cell2 = tf.contrib.legacy_seq2seq.rnn_decoder(outs_cell1, self.istate_cell2, self.cell2, loop_function=None, scope='cell2')

	# ----- start building the Mixture Density Network on top (start with a dense layer to predict the MDN params)
		#out_cell0 = tf.reshape(tf.concat(outs_cell0, 1), [-1, args.rnn_size]) 
		#dense1 = tf.layers.dense(inputs=out_cell0, units=1024, activation=tf.nn.relu)
		#dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
		#dense3 = tf.layers.dense(inputs=dense2, units=args.rnn_size, activation=tf.nn.relu)

		n_out = self.nmixtures * 6 * self.pjoints# params = end_of_stroke + 6 parameters per Gaussian
		#n_out = 40
		#print n_out
		with tf.variable_scope('mdn_dense'):
			mdn_w = tf.get_variable("output_w", [self.rnn_size, n_out], initializer=self.graves_initializer)
			mdn_b = tf.get_variable("output_b", [n_out], initializer=self.graves_initializer)
			#print mdn_w, mdn_b
		out_cell1 = tf.reshape(tf.concat(outs_cell2, 1), [-1, args.rnn_size]) #concat outputs for efficiency

		#out_cell2 = tf.reshape(tf.concat(dense2, 1), [960, args.rnn_size]) #concat outputs for efficiency
		#print out_cell2
		output = tf.nn.xw_plus_b(out_cell1, mdn_w, mdn_b) #data flows through dense nn


	# ----- build mixture density cap on top of second recurrent cell
		def gaussian2d(x1, x2, mu1, mu2, s1, s2, rho):
			# define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
	
			x_mu1 = tf.subtract(x1, mu1)
			x_mu2 = tf.subtract(x2, mu2)
			Z = tf.square(tf.div(x_mu1, s1)) + \
			    tf.square(tf.div(x_mu2, s2)) - \
			    2*tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
			rho_square_term = 1-tf.square(rho)
			power_e = tf.exp(tf.div(-Z,2*rho_square_term))
			regularize_term = 2*np.pi*tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
			gaussian = tf.div(power_e, regularize_term)
			return gaussian

		def get_loss(pi, x1_data, x2_data, mu1, mu2, sigma1, sigma2, rho):
			# define loss function (eq 26 of http://arxiv.org/abs/1308.0850)
			'''
			gaussian = gaussian2d(x1_data, x2_data, mu1, mu2, sigma1, sigma2, rho)
			term1 = tf.multiply(gaussian, pi)
			term1 = tf.reduce_sum(term1, 1, keep_dims=True) #do inner summation
			term1 = -tf.log(tf.maximum(term1, 1e-20)) # some errors are zero -> numerical errors.

			return tf.reduce_sum(term1) #do outer summation
			'''

			mus1 = tf.split(mu1, self.nmixtures, 1)
			mus2 = tf.split(mu2, self.nmixtures, 1)

			sigmas1 = tf.split(sigma1, self.nmixtures, 1) 
			sigmas2 = tf.split(sigma2, self.nmixtures, 1)

			rhos = tf.split(rho, self.nmixtures, 1)

			pis = tf.split(pi, self.nmixtures, 1)

			for i in range(self.nmixtures):
				gaussian = gaussian2d(x1_data, x2_data, mus1[i], mus2[i], sigmas1[i], sigmas2[i], rhos[i])
				#print gaussian
				try:
					term1 += tf.multiply(gaussian, pis[i])
				except:
					term1 = tf.multiply(gaussian, pis[i])
			#term1 = tf.reduce_sum(term1, 1, keep_dims=True) #do inner summation
			term1 = -tf.log(tf.maximum(term1, 1e-20)) # some errors are zero -> numerical errors.
			return tf.reduce_sum(term1) #do outer summation

		# now transform dense NN outputs into params for MDN
		def get_mdn_coef(Z):
			# returns the tf slices containing mdn dist params (eq 18...23 of http://arxiv.org/abs/1308.0850)
			pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(Z, 6, 1)
			self.pi_hat, self.sigma1_hat, self.sigma2_hat, self.rho_hat = \
										pi_hat, sigma1_hat, sigma2_hat, rho_hat # these are useful for bias method during sampling

			pi = tf.nn.softmax(pi_hat) # softmax z_pi:
			mu1 = mu1_hat; mu2 = mu2_hat # leave mu1, mu2 as they are
			sigma1 = tf.exp(sigma1_hat); sigma2 = tf.exp(sigma2_hat) # exp for sigmas
			rho = tf.tanh(rho_hat) # tanh for rho (squish between -1 and 1)rrr

			return [pi, mu1, mu2, sigma1, sigma2, rho]

		# reshape target data (as we did the input data)
		#print self.target_data
		flat_target_data = tf.reshape(self.target_data,(self.tsteps*self.batch_size, 40))
		[x1_data, x2_data] = tf.split(flat_target_data, self.ndimensions, 1) #we might as well split these now
		[self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho] = get_mdn_coef(output)

		#loss = tf.losses.mean_squared_error(flat_target_data, output)
		#self.output = output

		loss = get_loss(self.pi, x1_data, x2_data, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho)
		self.cost = loss / (self.batch_size * self.tsteps)

		# ----- bring together all variables and prepare for training
		self.learning_rate = tf.Variable(0.0, trainable=False)
		self.decay = tf.Variable(0.0, trainable=False)
		self.momentum = tf.Variable(0.0, trainable=False)

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)

		if args.optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		elif args.optimizer == 'rmsprop':
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)
		else:
			raise ValueError("Optimizer type not recognized")
		self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

		# ----- some TensorFlow I/O
		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver(tf.global_variables())
		self.sess.run(tf.global_variables_initializer())

		# ----- for restoring previous models
	def try_load_model(self, save_path):
		load_was_success = True # yes, I'm being optimistic
		global_step = 0
		try:
			save_dir = '/'.join(save_path.split('/')[:-1])
			ckpt = tf.train.get_checkpoint_state(save_dir)
			load_path = ckpt.model_checkpoint_path
			self.saver.restore(self.sess, load_path)
		except:
			print("no saved model to load. starting new session")
			load_was_success = False
		else:
			print("loaded model: {}".format(load_path))
			self.saver = tf.train.Saver(tf.global_variables())
			global_step = int(load_path.split('-')[-1])
		return load_was_success, global_step
