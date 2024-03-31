import argparse
import json
import numpy as np
import csv

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('-c', '--conf', dest='conf')
args = parser.parse_args()
with open(args.conf, 'r', encoding='utf-8') as f:
    conf = json.load(f)

resource = np.random.randint(1, conf["resource_max"], conf["no_models"])

with open('resources_No_'+str(conf["no_models"])+'_max_'+str(conf["no_models"])+'.csv', mode='a+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(resource)