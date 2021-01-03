import yaml
import sys 
with open('../parameters/film2d.yaml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)
print(parameters)
data = sys.argv[1]):
print(data)