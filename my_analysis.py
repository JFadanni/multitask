import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from task import generate_trials, rule_name
from network import Model
import tools



allrules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

#allrules = ['fdgo', 'reactgo', 'fdanti', 'reactanti', 'delayanti']
#, 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']


for r in allrules:

    model_dir = '../new_train1/'+r
    rule = r

    model = Model(model_dir)
    hp = model.hp


    with tf.compat.v1.Session() as sess:
        model.restore()

        #trial = generate_trials(rule, hp, mode='test',batch_size = 1)
        trial = generate_trials(rule, hp, mode='random',batch_size = 200, initial_time=30)
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # All matrices have shape (n_time, n_condition, n_neuron)
        
        print(np.shape(trial.x))


        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get name of each variable
        names  = [var.name for var in var_list]


    # Take only the one example trial
    i_trial = 0

    '''
    for activity, title in zip([trial.x, h, y_hat],
                            ['input', 'recurrent', 'output']):
        
        plt.figure()
        plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot',
                   interpolation='none', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()
    '''

    print( trial.x.shape, trial.y.shape )

    x_train = np.vstack(np.swapaxes(trial.x, 0, 1))
    y_train = np.vstack(np.swapaxes(trial.y, 0, 1))


    print( np.shape(x_train), np.shape(y_train ))
    
    np.savetxt(model_dir+'/x_train.dat', x_train)
    np.savetxt(model_dir+'/y_train.dat', y_train)
    
    
    '''
    for param, name in zip(params, names):
        if len(param.shape) != 2:
            continue

        vmax = np.max(abs(param))*0.7
        plt.figure()
        # notice the transpose
        plt.imshow(param.T, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax,
               y_train.txt    interpolation='none', origin='lower')
        plt.title(name)
        plt.colorbar()
        plt.xlabel('From')
        plt.ylabel('To')
        plt.show()
    '''

    Nin=66

    w_rnn = params[0]
    w_in = w_rnn[:Nin]
    w_rec = w_rnn[Nin:]
    w_out = params[2]
    brec = params[1]
    bout = params[3]
    
    np.savetxt(model_dir+'/RNN_all_win.dat', w_in)
    np.savetxt(model_dir+'/RNN_all_wrec.dat', w_rec)
    np.savetxt(model_dir+'/RNN_all_wout.dat', w_out)

    np.savetxt(model_dir+'/RNN_all_brec.dat', brec)
    np.savetxt(model_dir+'/RNN_all_bout.dat', bout)
    
    
