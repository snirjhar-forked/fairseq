import pandas as pd
import os
import sys
import csv

def read_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    train_data = {}
    valid_data = {}
    
    for line in lines:
        if not line.startswith('epoch'):
            continue
        
        entry = [e.strip().split() for e in  line.strip().split('|')]
        
        epoch = int(entry[0][1])
        entry.pop(0)
        
        if entry[0][0] == 'valid':
            entry.pop(0)
            update_data = valid_data
        else:
            update_data = train_data
        
        data = {e[0]: float(e[1]) for e in entry}
        update_data[epoch] = data
    
    train_data = pd.DataFrame(train_data).T
    valid_data = pd.DataFrame(valid_data).T
    return train_data, valid_data

def save2csv(data, filename):
    content = ';\n'.join(data.to_csv(index_label='epoch',quoting=csv.QUOTE_NONNUMERIC,quotechar="'").splitlines())
    with open(filename, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    filename = sys.argv[1]
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'log.log')
    
    train_data, valid_data = read_log(filename)
    
    save2csv(train_data, filename.replace('.log', '_train.csv'))
    # train_data.to_excel(filename.replace('.log', '_train.xlsx'))
    # train_data.to_pickle(train_data, filename.replace('.log', '_train.pkl'))
    
    save2csv(valid_data, filename.replace('.log', '_valid.csv'))
    # valid_data.to_excel(filename.replace('.log', '_valid.xlsx'))
    # valid_data.to_pickle(valid_data, filename.replace('.log', '_valid.pkl'))

