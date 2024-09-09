import train2 as train
import json
import numpy as np

#allrules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

#allrules= ["fdgo"]
allrules= ["caudorostral"]

#allrules = ['delaygo', 'dm1', 'dm2']
#allrules = ['contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

for r in allrules:
    ruleset=r

    model_dir = '../new_train/'+r

#    with open(model_dir+'/log.json', 'r') as f:
#        logdata = json.load(f)

#    perf=logdata['perf_'+r]

    #print(r,np.mean(perf[-10:]))
    
    #print(ruleset)

    #print(model_dir)

      #    if(np.mean(perf[-10:])<0.99):

#        print('not good enough')
train.train(model_dir=model_dir, hp={'learning_rate': 0.001, 'activation': 'softplus',"target_perf" :0.99}, ruleset=ruleset, display_step=50,max_steps=2E6)
#    else: 
#        print(r,'good enough')


#train.train(model_dir='debug', hp={'learning_rate': 0.001}, ruleset='mante')
