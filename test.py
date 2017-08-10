from __future__ import print_function

import yaml
import numpy as np
import re
import random

doc = '''

colors:

    - &COLOR_BLACK 0,0,0
    - &COLOR_WHITE           255, 255, 255
    - &FOO
        x: 10
        y: 10

a: &a
 a1: *COLOR_BLACK
 a2: $b1

b: !sun
 <<: *a
 a1: *COLOR_WHITE
 b1: !random_color

'''

class Sun(object):

    def __init__(self, a1, a2, b1):
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        print('sun __init__', a1, a2, b1)


def color_constructor(loader, node):
    value = np.uint8([ int(s.strip()) for s in loader.construct_scalar(node).split(',') ])
    print('color value', value)
    return value

class Ref(object):

    def __init__(self, target):
        self.__target = target

    @property
    def target(self):
        return self.__target


def ref_constructor(loader, node):
    value = loader.construct_scalar(node)
    name = value[1:]
    return Ref(name)


def mapping_constructor_fn(fn):

    def mapping_constructor(loader, node):
        mapping = loader.construct_mapping(node, deep = True)
        resolve_refs([mapping])
        return fn(**mapping)

    return mapping_constructor


class Factory(object):

    def __init__(self):
        pass

    def __call__(self):
        return None

    @classmethod
    def apply(current, context_stack = []):
        for k, v in current.iteritems():
            if isinstance(v, dict):
                context_stack.append(v)
                Factory.apply(context_stack)
                context_stack.pop()
            elif isinstance(v, Ref):
                found = False
                for context in reversed(context_stack):
                    if v.target in context:
                        current[k] = context[v.target]
                        found = True
                        break
                if not found:
                    raise RuntimeError('Referenced property not found: ' + v.target)



class ValueFactory(Factory):

    def __init__(self, target):
        super(ValueFactory, self).__init__()
        self.__target = target

    def __call__(self, context_stack):
        for context in reversed(context_stack):
            if v.target in context:
                return context[v.target]
        raise RuntimeError('Referenced property not found: ' + v.target)



class ObjectFactory(Factory):

    def __init__(self, fn, kwargs):
        super(ObjectFactory, self).__init__()
        self.__fn = fn
        self.__kwargs = kwargs

    def __call__(self):
        return self.__fn(**self.__kwargs)

    @classmethod
    def for_class(cls, fn):

        def object_factory_constructor(loader, node):
            mapping = loader.construct_mapping(node, deep = True)
            return ObjectFactory(fn, mapping)

        return object_factory_constructor


class ObjectFactoryFactory(Factory):

    def __init__(self, fn, kwargs):
        super(ObjectFactoryFactory, self).__init__()

    def __call__(self):
        return self.__fn(**self.__kwargs)

    @classmethod
    def for_class(cls, fn):

        def object_factory_constructor(loader, node):
            mapping = loader.construct_mapping(node, deep = True)
            return ObjectFactory(fn, mapping)

        return object_factory_constructor

class RandomColorFactory(Factory):

    def __init__(self):
        super(RandomColorFactory, self).__init__()

    def __call__(self):
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


def random_color_constructor(loader, node):
    #arg = loader.construct_scalar(node)
    return RandomColorFactory()


yaml.add_constructor('!sun', ObjectFactory.for_class(Sun))
yaml.add_constructor('!color', color_constructor)
yaml.add_constructor('!ref', ref_constructor)
yaml.add_constructor('!random_color', random_color_constructor)
yaml.add_implicit_resolver('!ref', re.compile(r'^\$\w+$'))
yaml.add_implicit_resolver('!color', re.compile(r'^\d+, *\d+, *\d+$'))

result = yaml.load(doc)

print('before', result)

new_result = apply_factories([result])

print('after', new_result)

sun1 = new_result['b']()
sun2 = new_result['b']()
