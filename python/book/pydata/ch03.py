#!/usr/bin/env python
# coding: utf-8

# # Built-in Data Structures, Functions, 

# ## Data Structures and Sequences

# ### Tuple

# In[ ]:


tup = 4, 5, 6
tup


# In[ ]:


nested_tup = (4, 5, 6), (7, 8)
nested_tup


# In[ ]:


tuple([4, 0, 2])
tup = tuple('string')
tup


# In[ ]:


tup[0]


# In[ ]:


tup = tuple(['foo', [1, 2], True])
tup[2] = False


# In[ ]:


tup[1].append(3)
tup


# In[ ]:


(4, None, 'foo') + (6, 0) + ('bar',)


# In[ ]:


('foo', 'bar') * 4


# #### Unpacking tuples

# In[ ]:


tup = (4, 5, 6)
a, b, c = tup
b


# In[ ]:


tup = 4, 5, (6, 7)
a, b, (c, d) = tup
d


# tmp = a
# a = b
# b = tmp

# In[ ]:


a, b = 1, 2
a
b
b, a = a, b
a
b


# In[ ]:


seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a, b, c))


# In[ ]:


values = 1, 2, 3, 4, 5
a, b, *rest = values
a, b
rest


# In[ ]:


a, b, *_ = values


# #### Tuple methods

# In[ ]:


a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# ### List

# In[ ]:


a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
b_list = list(tup)
b_list
b_list[1] = 'peekaboo'
b_list


# In[ ]:


gen = range(10)
gen
list(gen)


# #### Adding and removing elements

# In[ ]:


b_list.append('dwarf')
b_list


# In[ ]:


b_list.insert(1, 'red')
b_list


# In[ ]:


b_list.pop(2)
b_list


# In[ ]:


b_list.append('foo')
b_list
b_list.remove('foo')
b_list


# In[ ]:


'dwarf' in b_list


# In[ ]:


'dwarf' not in b_list


# #### Concatenating and combining lists

# In[ ]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# In[ ]:


x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x


# everything = []
# for chunk in list_of_lists:
#     everything.extend(chunk)

# everything = []
# for chunk in list_of_lists:
#     everything = everything + chunk

# #### Sorting

# In[ ]:


a = [7, 2, 5, 1, 3]
a.sort()
a


# In[ ]:


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b


# #### Binary search and maintaining a sorted list

# In[ ]:


import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 2)
bisect.bisect(c, 5)
bisect.insort(c, 6)
c


# #### Slicing

# In[ ]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]


# In[ ]:


seq[3:4] = [6, 3]
seq


# In[ ]:


seq[:5]
seq[3:]


# In[ ]:


seq[-4:]
seq[-6:-2]


# In[ ]:


seq[::2]


# In[ ]:


seq[::-1]


# ### Built-in Sequence Functions

# #### enumerate

# i = 0
# for value in collection:
#    # do something with value
#    i += 1

# for i, value in enumerate(collection):
#    # do something with value

# In[ ]:


some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i
mapping


# #### sorted

# In[ ]:


sorted([7, 1, 2, 6, 0, 3, 2])
sorted('horse race')


# #### zip

# In[ ]:


seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
list(zipped)


# In[ ]:


seq3 = [False, True]
list(zip(seq1, seq2, seq3))


# In[ ]:


for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('{0}: {1}, {2}'.format(i, a, b))


# In[ ]:


pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),
            ('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
first_names
last_names


# #### reversed

# In[ ]:


list(reversed(range(10)))


# ### dict

# In[ ]:


empty_dict = {}
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
d1


# In[ ]:


d1[7] = 'an integer'
d1
d1['b']


# In[ ]:


'b' in d1


# In[ ]:


d1[5] = 'some value'
d1
d1['dummy'] = 'another value'
d1
del d1[5]
d1
ret = d1.pop('dummy')
ret
d1


# In[ ]:


list(d1.keys())
list(d1.values())


# In[ ]:


d1.update({'b' : 'foo', 'c' : 12})
d1


# #### Creating dicts from sequences

# mapping = {}
# for key, value in zip(key_list, value_list):
#     mapping[key] = value

# In[ ]:


mapping = dict(zip(range(5), reversed(range(5))))
mapping


# #### Default values

# if key in some_dict:
#     value = some_dict[key]
# else:
#     value = default_value

# value = some_dict.get(key, default_value)

# In[ ]:


words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
by_letter


# for word in words:
#     letter = word[0]
#     by_letter.setdefault(letter, []).append(word)

# from collections import defaultdict
# by_letter = defaultdict(list)
# for word in words:
#     by_letter[word[0]].append(word)

# #### Valid dict key types

# In[ ]:


hash('string')
hash((1, 2, (2, 3)))
hash((1, 2, [2, 3])) # fails because lists are mutable


# In[ ]:


d = {}
d[tuple([1, 2, 3])] = 5
d


# ### set

# In[ ]:


set([2, 2, 2, 1, 3, 3])
{2, 2, 2, 1, 3, 3}


# In[ ]:


a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}


# In[ ]:


a.union(b)
a | b


# In[ ]:


a.intersection(b)
a & b


# In[ ]:


c = a.copy()
c |= b
c
d = a.copy()
d &= b
d


# In[ ]:


my_data = [1, 2, 3, 4]
my_set = {tuple(my_data)}
my_set


# In[ ]:


a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)
a_set.issuperset({1, 2, 3})


# In[ ]:


{1, 2, 3} == {3, 2, 1}


# ### List, Set, and Dict Comprehensions

# [

# result = []
# for val in collection:
#     if 

# In[ ]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]


# dict_comp = {

# set_comp = {

# In[ ]:


unique_lengths = {len(x) for x in strings}
unique_lengths


# In[ ]:


set(map(len, strings))


# In[ ]:


loc_mapping = {val : index for index, val in enumerate(strings)}
loc_mapping


# #### Nested list comprehensions

# In[ ]:


all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
            ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]


# names_of_interest = []
# for names in all_data:
#     enough_es = [name for name in names if name.count('e') >= 2]
#     names_of_interest.extend(enough_es)

# In[ ]:


result = [name for names in all_data for name in names
          if name.count('e') >= 2]
result


# In[ ]:


some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened


# flattened = []
# 
# for tup in some_tuples:
#     for x in tup:
#         flattened.append(x)

# In[ ]:


[[x for x in tup] for tup in some_tuples]


# ## Functions

# def my_function(x, y, z=1.5):
#     if z > 1:
#         return z * (x + y)
#     else:
#         return z / (x + y)

# my_function(5, 6, z=0.7)
# my_function(3.14, 7, 3.5)
# my_function(10, 20)

# ### Namespaces, Scope, and Local Functions

# def func():
#     a = []
#     for i in range(5):
#         a.append(i)

# a = []
# def func():
#     for i in range(5):
#         a.append(i)

# In[ ]:


a = None
def bind_a_variable():
    global a
    a = []
bind_a_variable()
print(a)


# ### Returning Multiple Values

# def f():
#     a = 5
#     b = 6
#     c = 7
#     return a, b, c
# 
# a, b, c = f()

# return_value = f()

# def f():
#     a = 5
#     b = 6
#     c = 7
#     return {'a' : a, 'b' : b, 'c' : c}

# ### Functions Are Objects

# In[ ]:


states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda',
          'south   carolina##', 'West virginia?']


# In[ ]:


import re

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result


# In[ ]:


clean_strings(states)


# In[ ]:


def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result


# In[ ]:


clean_strings(states, clean_ops)


# In[ ]:


for x in map(remove_punctuation, states):
    print(x)


# ### Anonymous (Lambda) Functions

# def short_function(x):
#     return x * 2
# 
# equiv_anon = lambda x: x * 2

# def apply_to_list(some_list, f):
#     return [f(x) for x in some_list]
# 
# ints = [4, 0, 1, 5, 6]
# apply_to_list(ints, lambda x: x * 2)

# In[ ]:


strings = ['foo', 'card', 'bar', 'aaaa', 'abab']


# In[ ]:


strings.sort(key=lambda x: len(set(list(x))))
strings


# ### Currying: Partial Argument Application

# def add_numbers(x, y):
#     return x + y

# add_five = lambda y: add_numbers(5, y)

# from functools import partial
# add_five = partial(add_numbers, 5)

# ### Generators

# In[ ]:


some_dict = {'a': 1, 'b': 2, 'c': 3}
for key in some_dict:
    print(key)


# In[ ]:


dict_iterator = iter(some_dict)
dict_iterator


# In[ ]:


list(dict_iterator)


# In[ ]:


def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n ** 2))
    for i in range(1, n + 1):
        yield i ** 2


# In[ ]:


gen = squares()
gen


# In[ ]:


for x in gen:
    print(x, end=' ')


# #### Generator expresssions

# In[ ]:


gen = (x ** 2 for x in range(100))
gen


# def _make_gen():
#     for x in range(100):
#         yield x ** 2
# gen = _make_gen()

# In[ ]:


sum(x ** 2 for x in range(100))
dict((i, i **2) for i in range(5))


# #### itertools module

# In[ ]:


import itertools
first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator


# ### Errors and Exception Handling

# In[ ]:


float('1.2345')
float('something')


# In[ ]:


def attempt_float(x):
    try:
        return float(x)
    except:
        return x


# In[ ]:


attempt_float('1.2345')
attempt_float('something')


# In[ ]:


float((1, 2))


# In[ ]:


def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x


# In[ ]:


attempt_float((1, 2))


# In[ ]:


def attempt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


# f = open(path, 'w')
# 
# try:
#     write_to_file(f)
# finally:
#     f.close()

# f = open(path, 'w')
# 
# try:
#     write_to_file(f)
# except:
#     print('Failed')
# else:
#     print('Succeeded')
# finally:
#     f.close()

# #### Exceptions in IPython

# In [10]: %run examples/ipython_bug.py
# ---------------------------------------------------------------------------
# AssertionError                            Traceback (most recent call last)
# /home/wesm/code/pydata-book/examples/ipython_bug.py in <module>()
#      13     throws_an_exception()
#      14
# ---> 15 calling_things()
# 
# /home/wesm/code/pydata-book/examples/ipython_bug.py in calling_things()
#      11 def calling_things():
#      12     works_fine()
# ---> 13     throws_an_exception()
#      14
#      15 calling_things()
# 
# /home/wesm/code/pydata-book/examples/ipython_bug.py in throws_an_exception()
#       7     a = 5
#       8     b = 6
# ----> 9     assert(a + b == 10)
#      10
#      11 def calling_things():
# 
# AssertionError:

# ## Files and the Operating System

# In[ ]:


get_ipython().magic(u'pushd book-materials')


# In[ ]:


path = 'examples/segismundo.txt'
f = open(path)


# for line in f:
#     pass

# In[ ]:


lines = [x.rstrip() for x in open(path)]
lines


# In[ ]:


f.close()


# In[ ]:


with open(path) as f:
    lines = [x.rstrip() for x in f]


# In[ ]:


f = open(path)
f.read(10)
f2 = open(path, 'rb')  # Binary mode
f2.read(10)


# In[ ]:


f.tell()
f2.tell()


# In[ ]:


import sys
sys.getdefaultencoding()


# In[ ]:


f.seek(3)
f.read(1)


# In[ ]:


f.close()
f2.close()


# In[ ]:


with open('tmp.txt', 'w') as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)
with open('tmp.txt') as f:
    lines = f.readlines()
lines


# In[ ]:


import os
os.remove('tmp.txt')


# ### Bytes and Unicode with Files

# In[ ]:


with open(path) as f:
    chars = f.read(10)
chars


# In[ ]:


with open(path, 'rb') as f:
    data = f.read(10)
data


# In[ ]:


data.decode('utf8')
data[:4].decode('utf8')


# In[ ]:


sink_path = 'sink.txt'
with open(path) as source:
    with open(sink_path, 'xt', encoding='iso-8859-1') as sink:
        sink.write(source.read())
with open(sink_path, encoding='iso-8859-1') as f:
    print(f.read(10))


# In[ ]:


os.remove(sink_path)


# In[ ]:


f = open(path)
f.read(5)
f.seek(4)
f.read(1)
f.close()


# In[ ]:


get_ipython().magic(u'popd')


# ## Conclusion
