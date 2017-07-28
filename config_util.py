
def process_config(config):
    config = __convert_config_value(config)
    config = __replace_config_value(config, [config])
    return config

def __replace_config_value(v, context_stack):

    if isinstance(v, basestring):

        for context in reversed(context_stack):
            if v in context:
                v = context[v]
            if not isinstance(v, basestring):
                break

    if isinstance(v, list):

        v = [ __replace_config_value(i, context_stack) for i in v ]

    elif isinstance(v, dict):

        context_stack.append(v)
        v = { k: __replace_config_value(v, context_stack) for k, v in v.iteritems() }
        context_stack.pop()

    return v

def __convert_config_value(v):

    if isinstance(v, basestring) and ',' in v:

        v = [ __convert_config_string_to_number(i) for i in v.split(',') ]

    if isinstance(v, list):

        v = [ __convert_config_value(i) for i in v ]

    elif isinstance(v, dict):

        v = { k: __convert_config_value(v) for k, v in v.iteritems() }

    return v


def __convert_config_string_to_number(s):
    s = s.strip()
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s

