import numpy as np
import tensorflow as tf

import argparse
import time
import os

from model import Model
from utils import *
from sample import *

def main():
	parser = argparse.ArgumentParser()

	#general model params
	parser.add_argument('--train', dest='train', action='store_true', help='train the model')
	parser.add_argument('--sample', dest='train', action='store_false', help='sample from the model')
	parser.add_argument('--rnn_size', type=int, default=500, help='size of RNN hidden state')
	parser.add_argument('--tsteps', type=int, default=200, help='RNN time steps (for backprop)')
	parser.add_argument('--nmixtures', type=int, default=10, help='number of gaussian mixtures')
	parser.add_argument('--pjoints', type=int, default=20, help='number of points (joints) to keep track of')
	parser.add_argument('--ndimensions', type=int, default=2, help='number of dimensions for joint positions')

	# window params
	parser.add_argument('--kmixtures', type=int, default=8, help='number of gaussian mixtures for character window')
	parser.add_argument('--alphabet', type=str, default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', \
						help='default is a-z, A-Z, space, and <UNK> tag')
	parser.add_argument('--tsteps_per_ascii', type=int, default=25, help='expected number of pen points per character')

	# training params
	parser.add_argument('--batch_size', type=int, default=10, help='batch size for each gradient step')
	parser.add_argument('--sequence_shift', type=int, default=50, help='batch size for each gradient step')
	parser.add_argument('--nbatches', type=int, default=500, help='number of batches per epoch')
	parser.add_argument('--nepochs', type=int, default=1000, help='number of epochs')
	parser.add_argument('--dropout', type=float, default=0.85, help='probability of keeping neuron during dropout')

	parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
	parser.add_argument('--optimizer', type=str, default='rmsprop', help="ctype of optimizer: 'rmsprop' 'adam'")
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--lr_decay', type=float, default=1.0, help='decay rate for learning rate')
	parser.add_argument('--decay', type=float, default=0.95, help='decay rate for rmsprop')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for rmsprop')

	#book-keeping
	parser.add_argument('--data_scale', type=int, default=50, help='amount to scale data down before training')
	parser.add_argument('--log_dir', type=str, default='./logs/', help='location, relative to execution, of log files')
	parser.add_argument('--data_dir', type=str, default='./data/normalized', help='location, relative to execution, of data')
	parser.add_argument('--save_path', type=str, default='saved/model.ckpt', help='location to save model')
	parser.add_argument('--save_every', type=int, default=500, help='number of batches between each save')

	#sampling
	parser.add_argument('--text', type=str, default='', help='string for sampling model (defaults to test cases)')
	parser.add_argument('--style', type=int, default=-1, help='optionally condition model on a preset style (using data in styles.p)')
	parser.add_argument('--bias', type=float, default=0.0, help='higher bias means neater, lower means more diverse (range is 0-5)')
	parser.add_argument('--sleep_time', type=int, default=60*5, help='time to sleep between running sampler')
	parser.set_defaults(train=True)
	args = parser.parse_args()



	train_model(args) if args.train else sample_model(args)

def train_model(args):
	logger = Logger(args) # make logging utility
	#logger.write("\nTRAINING MODE...")
	#logger.write("{}\n".format(args))
	#logger.write("loading data...")
	data_loader = DataLoader(args, logger=logger)
	
	#logger.write("building model...")
	model = Model(args, logger=logger)

	#logger.write("attempt to load saved model...")
	load_was_success, global_step = model.try_load_model(args.save_path)

	v_x, v_y = data_loader.validation_data()
	valid_inputs = {model.input_data: v_x, model.target_data: v_y}

	#logger.write("training...")
	model.sess.run(tf.assign(model.decay, args.decay ))
	model.sess.run(tf.assign(model.momentum, args.momentum ))
	running_average = 0.0 ; remember_rate = 0.99
	for e in range(global_step/args.nbatches, args.nepochs):
		model.sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.lr_decay ** e)))
		#logger.write("learning rate: {}".format(model.learning_rate.eval()))

		#c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
		#h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()

		for b in range(global_step%args.nbatches, args.nbatches):
			
			i = e * args.nbatches + b
			if global_step is not 0 : i+=1 ; global_step = 0

			if i % args.save_every == 0 and (i > 0):
				model.saver.save(model.sess, args.save_path, global_step = i) ; #logger.write('SAVED MODEL')

			start = time.time()
			x, y = data_loader.next_batch()

			feed = {model.input_data: x, model.target_data: y}

			[train_loss, _] = model.sess.run([model.cost, model.train_op], feed)
			feed.update(valid_inputs)
			[valid_loss] = model.sess.run([model.cost], feed)
			
			running_average = running_average*remember_rate + train_loss*(1-remember_rate)
			end = time.time()
			if i % 20 is 0: 
				logger.write("{}/{}, loss = {:.3f}, regloss = {:.5f}, valid_loss = {:.3f}, time = {:.3f}, learning_rate={:.10f}" \
				.format(i, args.nepochs * args.nbatches, train_loss, running_average, valid_loss, end - start, args.learning_rate) )


def sample_model(args, logger=None):


	logger = Logger(args) if logger is None else logger # instantiate logger, if None
	logger.write("\nSAMPLING MODE...")
	logger.write("loading data...")
	
	logger.write("building model...")
	model = Model(args, logger)

	logger.write("attempt to load saved model...")
	load_was_success, global_step = model.try_load_model(args.save_path)

	if load_was_success:
		sample(model, args)
	else:
		logger.write("load failed, sampling canceled")



if __name__ == '__main__':
	main()
