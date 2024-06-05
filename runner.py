import os

lr = [1e-6,2e-6,3e-6,1e-5,2e-5]
dec = [0.01,0.02,0.005,0.1]
warm = [0.1,0.2,0.05]

for i in range(len(lr)):
    for j in range(len(dec)):
        for k in range(len(dec)):
            os.system(f'python3 statement_tune.py --Exp_name "XLMR-{i}{j}{k}" --transformer "xlm-roberta-base" --save "./XLMR-{i}{j}{k}" --tr_ep 4 --tr_batch 16 --lr lr[i] --decay dec[j] --warmup warm[k]')
            

