import os

lr = [1e-6,1e-7,2e-7]
warm = [0.1,0.15,0.2]

for i in range(len(lr)):
    for j in range(len(warm)):
        os.system(f'python3 statement_tune.py --Exp_name "XLMR-{i}{j}{k}" --transformer "xlm-roberta-base" --save "./XLMR-{i}{j}{k}" --tr_ep 4 --tr_batch 16 --lr {lr[i]} --warmup {warm[j]}')
            


            

