import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from task import generate_trials, rule_name
from network import Model
import tools



allrules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

#allrules = ['fdgo', 'reactgo', 'fdanti', 'reactanti', 'delayanti']
#, 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
allrules = ['reactgo']

for r in allrules:

    model_dir = '../new_train1/'+r
    rule = r

    model = Model(model_dir)
    hp = model.hp

    if r == "reactgo":
        initial_time = 60
    else:
        initial_time = 30
    with tf.compat.v1.Session() as sess:
        model.restore()

        #trial = generate_trials(rule, hp, mode='test',batch_size = 1)
        trial = generate_trials(rule, hp, mode='random',batch_size = 200, initial_time=initial_time)
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

    
    for activity, title in zip([trial.x, h, y_hat],
                            ['input', 'recurrent', 'output']):
        
        plt.figure()
        plt.imshow(activity[initial_time:,i_trial,:].T, aspect='auto', cmap='hot',
                   interpolation='none', origin='lower')
        plt.suptitle(rule)
        plt.title(title)
        plt.colorbar()
        plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(y_hat[initial_time:,i_trial,:])
    T = y_hat.shape[0]
    print(T)
    ax.axvline(T//2)
    ax.axvline((T//5)*4)
    plt.show()



    print( trial.x.shape, trial.y.shape )

    x_train = np.vstack(np.swapaxes(trial.x[initial_time-30:], 0, 1))
    y_train = np.vstack(np.swapaxes(trial.y[initial_time-30:], 0, 1))


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
    
    print(50*'-')
    print(len(params))
    for i,p in enumerate(params):
        print(i,":", np.shape(p))
#    w_in = params[0]
#    w_rec = params[3]
#    w_out = params[5]
#    brec = params[4]
#    bout = params[6]

    w_rnn = params[0]
    w_in = w_rnn[:Nin]
    w_rec = w_rnn[Nin:]
    w_out = params[2]
    brec = params[1]
    bout = params[3]
    print(w_in.shape, w_rec.shape, w_out.shape, brec.shape, bout.shape)

    
    np.savetxt(model_dir+'/RNN_all_win.dat', w_in)
    np.savetxt(model_dir+'/RNN_all_wrec.dat', w_rec)
    np.savetxt(model_dir+'/RNN_all_wout.dat', w_out)

    np.savetxt(model_dir+'/RNN_all_brec.dat', brec)
    np.savetxt(model_dir+'/RNN_all_bout.dat', bout)
    
    
