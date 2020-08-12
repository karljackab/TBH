import sys
import yaml

def read_config(display_config=True):
    """
    parse argv and read it as config
    """
    if len(sys.argv) < 2:
        print('ERROR: config name must be specified')
        print('Usage: python3 retrieve.py $config_name')
        exit(1)

    with open(f'config/{sys.argv[1]}.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if display_config:
        print('============================')
        print('========== Config ==========')
        for key in config:
            print(f'{key}:\t{config[key]}')
        print('============================')
    
    return config