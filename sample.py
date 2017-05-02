import numpy as np
import tensorflow as tf
import cPickle as pickle
import sys
import csv

from utils import *

def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)

    return x[0][0], x[0][1]

def get_style_states(model, args):
    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
    if args.style is -1: return [c0, c1, c2, h0, h1, h2] #model 'chooses' random style

    with open(os.path.join(args.data_dir, 'styles.p'),'r') as f:
        style_strokes, style_strings = pickle.load(f)

    style_strokes, style_string = style_strokes[args.style], style_strings[args.style]
    style_onehot = [to_one_hot(style_string, model.ascii_steps, args.alphabet)]
        
    style_stroke = np.zeros((1, 1, 3), dtype=np.float32)
    style_kappa = np.zeros((1, args.kmixtures, 1))
    prime_len = 500 # must be <= 700
    
    for i in xrange(prime_len):
        style_stroke[0][0] = style_strokes[i,:]
        feed = {model.input_data: style_stroke, model.char_seq: style_onehot, model.init_kappa: style_kappa, \
                model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
                model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}
        fetch = [model.new_kappa, \
                 model.fstate_cell0.c, model.fstate_cell1.c, model.fstate_cell2.c,
                 model.fstate_cell0.h, model.fstate_cell1.h, model.fstate_cell2.h]
        [style_kappa, c0, c1, c2, h0, h1, h2] = model.sess.run(fetch, feed)
    return [c0, c1, c2, np.zeros_like(h0), np.zeros_like(h1), np.zeros_like(h2)] #only the c vectors should be primed

def sample(model, args):
    # initialize some parameters
    #one_hot = [to_one_hot(input_text, model.ascii_steps, args.alphabet)]         # convert input string to one-hot vector
    #[c0, c1, c2, h0, h1, h2] = get_style_states(model, args) # get numpy zeros states for all three LSTMs
    #kappa = np.zeros((1, args.kmixtures, 1))   # attention mechanism's read head should start at index 0
    csvfile = open('predictions.csv', 'a+')
    csvwriter = csv.writer(csvfile, delimiter=",")
    csvwriter.writerow(["joint_position", "x", "y"])
    dance_data = []
    for dance_file in os.listdir("data/normalized"):
        dance_file = "minmaxnormalized_kinect_skeleton03_29_17_21_09.csv"
        csvfile = open("data/normalized" + "/" + dance_file,"rb")

        print dance_file
        dancereader = csv.reader(csvfile, delimiter=',')  
        count = 0
        x_row = []
        y_row = []
        for row in dancereader:
            if count > 0:
                x_row.append(row[8]) 
                y_row.append(row[9])
                #csvwriter.writerow([count-1, row[8], row[9]])
                #print row[8], row[9]
            if count == 20:
                dance_data = np.concatenate((np.array(x_row), np.array(y_row)), axis=0)
                break
                
            count += 1
        break

        
    dance_data = np.reshape(np.array(dance_data), (1, 1,40))

    print dance_data
    prev_x = dance_data


    finished = False ; z = 0

    while not finished:
        if z %20 == 0:
            print "progress: " + str(z)
        '''
        feed = {model.input_data: prev_x}
        fetch = [model.output]
        [output] = model.sess.run(fetch, feed)
        #print output
        writetofile = np.reshape(output, (20, 2))
        #print writetofile
        for i in range(20):
            csvwriter.writerow([i, writetofile[i, 0], writetofile[i, 1]])
        '''
        feed = {model.input_data: prev_x}
        fetch = [model.pi_hat, model.mu1, model.mu2, model.sigma1_hat, model.sigma2_hat, model.rho_hat]
        [pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat] = model.sess.run(fetch, feed)
        #bias stuff:
        sigma1 = np.exp(sigma1_hat - args.bias) ; sigma2 = np.exp(sigma2_hat - args.bias)
        #pi_hat *= 1 + args.bias # apply bias

        shape = (10, 20)
        #pi_hat = np.transpose(np.reshape(pi_hat, (shape)))
        pi_hat = np.reshape(pi_hat, shape)
        pi = np.zeros_like(pi_hat) # need to preallocate
        
        pi = np.exp(pi_hat)/  np.sum(np.exp(pi_hat), axis=0) # softmax
        mu1 = np.reshape(mu1, shape)
        mu2 = np.reshape(mu2, shape)
        sigma1 = np.reshape(sigma1, shape)
        sigma2 = np.reshape(sigma2, shape)
        rho = np.reshape(np.tanh(rho_hat), shape)
        next = []
        for i in range(20):
            # choose a component from the MDN
            idx = np.random.choice(pi.shape[0], p=pi[:,i])
            #output = sample_gaussian2d(mu1[i][idx], mu2[i][idx], sigma1[i][idx], sigma2[i][idx], rho[i][idx])
            output = sample_gaussian2d(mu1[idx][i], mu2[idx][i], sigma1[idx][i], sigma2[idx][i], rho[idx][i])

            next.append([output[0], output[1]])
            #print mu1[idx][i], mu2[idx][i]
            csvwriter.writerow([i, mu1[idx][i], mu2[idx][i]])

        
        # test if finished (has the read head seen the whole ascii sequence?)
        # main_kappa_idx = np.where(alpha[0]==np.max(alpha[0]));
        # finished = True if kappa[0][main_k   appa_idx] > len(input_text) else False

        finished = True if z > 150 else False
        # new input is previous output
        prev_x = np.reshape(np.array(next), (1, 1,40))
        #print prev_x
        #prev_x = np.reshape(output, (1, 1,40))
        z+=1


# plots parameters from the attention mechanism
def window_plots(phis, windows, save_path='.'):
    import matplotlib.cm as cm
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.title('Phis', fontsize=20)
    plt.xlabel("ascii #", fontsize=15)
    plt.ylabel("time steps", fontsize=15)
    plt.imshow(phis, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.subplot(122)
    plt.title('Soft attention window', fontsize=20)
    plt.xlabel("one-hot vector", fontsize=15)
    plt.ylabel("time steps", fontsize=15)
    plt.imshow(windows, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

# a heatmap for the probabilities of each pen point in the sequence
def gauss_plot(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib.mlab as mlab
    import matplotlib.cm as cm
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize) #
    buff = 1 ; epsilon = 1e-4
    minx, maxx = np.min(strokes[:,0])-buff, np.max(strokes[:,0])+buff
    miny, maxy = np.min(strokes[:,1])-buff, np.max(strokes[:,1])+buff
    delta = abs(maxx-minx)/400. ;

    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(strokes.shape[0]):
        gauss = mlab.bivariate_normal(X, Y, mux=strokes[i,0], muy=strokes[i,1], \
            sigmax=strokes[i,2], sigmay=strokes[i,3], sigmaxy=0) # sigmaxy=strokes[i,4] gives error
        Z += gauss/(np.max(gauss) + epsilon)

    plt.title(title, fontsize=20)
    plt.imshow(Z)
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

# plots the stroke data (handwriting!)
def line_plot(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()
