import train2 as train
import json
import numpy as np

allrules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
#allrules = ['reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

#allrules= ["fdgo"]
#allrules= ["reactgo"]
#allrules= ["caudorostral"]

#allrules = ['delaygo', 'dm1', 'dm2']
#allrules = ['contextdm1', 'contextdm2', 'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

for r in allrules:
    ruleset=r

    model_dir = '../new_train1/'+r

    try:
        with open(model_dir+'/hp.json', 'r') as f:
            hp = json.load(f)
    except:
        hp = {}

    print(hp)


#    with open(model_dir+'/log.json', 'r') as f:
#        logdata
#    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1. = json.load(f)

#    perf=logdata['perf_'+r]

    #print(r,np.mean(perf[-10:]))
    
    #print(ruleset)

    #print(model_dir)

      #    if(np.mean(perf[-10:])<0.99):

#        print('not good enough')
    #train.train(model_dir=model_dir, hp={'learning_rate': 0.001, 'activation': 'softplus'}, ruleset=ruleset, display_step=50,max_steps=2E6)
    #train.train(model_dir=model_dir, hp=hp, ruleset=ruleset, display_step=50,max_steps=2E6,trainables="no_input")
    #train.train(model_dir=model_dir, hp=hp, ruleset=ruleset, display_step=50,max_steps=2E6,trainables="all")
    train.train(model_dir=model_dir, hp={'learning_rate': 0.001, 'activation': 'softplus',"target_perf" :0.999}, ruleset=ruleset, display_step=50,max_steps=2E6,trainables="all")
#    else: 
#        print(r,'good enough')


#train.train(model_dir='debug', hp={'learning_rate': 0.001}, ruleset='mante')
