from __future__ import division
from __future__ import print_function

import yaml

def process_config(config):
    config = convert_config_value(config)
    config = replace_config_value(config, [config])
    return config

def replace_config_value(v, context_stack):

    if isinstance(v, basestring):

        for context in reversed(context_stack):
            if v in context:
                v = context[v]
            if not isinstance(v, basestring):
                break

    if isinstance(v, list):

        v = [ replace_config_value(i, context_stack) for i in v ]

    elif isinstance(v, dict):

        context_stack.append(v)
        v = { k: replace_config_value(v, context_stack) for k, v in v.iteritems() }
        context_stack.pop()

    return v

def convert_config_value(v):

    if isinstance(v, basestring) and ',' in v:

        v = [ convert_config_string_to_number(i) for i in v.split(',') ]

    if isinstance(v, list):

        v = [ convert_config_value(i) for i in v ]

    elif isinstance(v, dict):

        v = { k: convert_config_value(v) for k, v in v.iteritems() }

    return v


def convert_config_string_to_number(s):
    s = s.strip()
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s


with open('/home/pi/pyhugm/config.yaml') as f:
    config = yaml.load(f)

print('=======')
print(config)
print('=======')
config = process_config(config)
print(config)
print('=======')
print(yaml.dump(config, default_flow_style=False))





