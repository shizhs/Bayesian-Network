'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Shizuka Hayashi    zID: z5165356

Name: Keshuo Lin    zID: z5244677
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries 
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle


###################################
# Code stub
# 
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
# 

# Markov networks based on the given information and office structure
graphX ={
    "c1": ["c1","c2","r7","r25"],
    "c2": ["c1","c2","c4","r34"],
    "c3": [],
    "c4": ["c2","c4","o1","r28","r29","r35"],
    "o1": [],
    "r1": [],
    "r2": ["r1","r2","r4"],
    "r3": ["r1","r3","r7"],
    "r4": ["r2","r4","r8"],
    "r5": [],
    "r6": ["c3","r5","r6"],
    "r7": ["c1","r3","r7"],
    "r8": ["r4","r8","r9"],
    "r9": ["r5","r8","r9","r13"],
    "r10": ["c3","r10"],
    "r11": ["c3","r11"],
    "r12": ["r12","r22"],
    "r13": ["r9","r13","r24"],
    "r14": ["r14","r24"],
    "r15": ["c3","r15"],
    "r16": [],
    "r17": ["c3","r17"],
    "r18": ["c3","r18"],
    "r19": ["c3","r19"],
    "r20": ["c3","r20"],
    "r21": ["c3","r21"],
    "r22": ["r12","r22","r25"],
    "r23": ["r23","r24"],
    "r24": [],
    "r25": [],
    "r26": ["r25","r26","r27"],
    "r27": ["r26","r27","r32"],
    "r28": ["c4","r28"],
    "r29": ["c4","r29","r30"],
    "r30": ["r29","r30"],
    "r31": [],
    "r32": ["r27","r31","r32","r33"],
    "r33": ["r32","r33"],
    "r34": ["c2","r34"],
    "r35": ["c4","r35"],
    "outside": ["r12","outside"]
}

graphE={
    "r16": ["reliable_sensor1"],
    "r5": ["reliable_sensor2"],
    "r25": ["reliable_sensor3"],
    "r31": ["reliable_sensor4"],
    "o1": ["unreliable_sensor1"],
    "c3": ["unreliable_sensor2"],
    "r1": ["unreliable_sensor3"],
    "r24": ["unreliable_sensor4"],
    "r8": ["door_sensor1"],
    "r9": ["door_sensor1"],
    "c1": ["door_sensor2"],
    "c2": ["door_sensor2"],
    "r26": ["door_sensor3"],
    "r27": ["door_sensor3"],
    "c4": ["door_sensor4"],
    "r35": ["door_sensor4"]
}

graph_min_E = {
    "r2":["unreliable_sensor3"],
    "r3":["unreliable_sensor3"],
    "r6":["reliable_sensor2"],
    'r10':['unreliable_sensor2'],
    'r15':['unreliable_sensor2'],
    'r11':['unreliable_sensor2'],
    'r12':['reliable_sensor3'],
    'r17':['unreliable_sensor2'],
    'r18':['unreliable_sensor2'],
    'r19':['unreliable_sensor2'],
    'r20':['unreliable_sensor2'],
    'r21':['unreliable_sensor2'],
    'r22':['reliable_sensor3'],
    'r13':['unreliable_sensor4'],
    'r14':['unreliable_sensor4'],
    'r23':['unreliable_sensor4'],
    'r32':['reliable_sensor4'],
    'r4':["unreliable_sensor3"],
    'r7':["unreliable_sensor3"],
    'r28':["unreliable_sensor1"],
    'r29':["unreliable_sensor1"],
    'r33':["reliable_sensor4"],
    'r30':['unreliable_sensor1'],
    'r34':['unreliable_sensor1'],
    'outside':['reliable_sensor3']
}

def transposeGraph(G):
    """
    argument 
    `G`, an adjacency list representation of a graph
    """      
    GT={}
    for v in G:
        for w in G[v]:
            if w in GT:
                GT[w].append(v)
            else:
                GT[w]=[v]
    return GT

graph_sensor=transposeGraph(graphE)
    
# load probability table from picke files
with open('emission.pickle', 'rb') as handle:
    emission = pickle.load(handle)

with open('transition.pickle', 'rb') as handle:
    transition = pickle.load(handle)
    
with open('reversed_emission.pickle', 'rb') as handle:
    rev_emission = pickle.load(handle)

with open('sensor_prob.pickle', 'rb') as handle:
    sensor_prob = pickle.load(handle)

# Helper functions
def normalize(f):
    """
    argument 
    `f`, factor to be normalized.
    
    Returns a new factor f' as a copy of f with entries that sum up to 1
    """ 
    new_f=f.copy()
    total=0
    for elem in new_f['table']:
        total+=2**new_f['table'][elem]
    for elem in new_f['table']:
        x=2**f['table'][elem]
        new_f['table'][elem]=math.log(x/total) if x!=0 else -math.inf
        
    return new_f

def calculateB(pred, prob, new_e):
    global B
    for key in prob:
        prob[key]=normalize(prob[key])
    for room in prob:
        if room in graphE:
            B[room]=emission[room]['table'][(pred[room], new_e[graphE[room][0]])]+prob[room]['table'][(pred[room],)]
        else:
            x=emission[room]['table'][(pred[room], new_e[graph_min_E[room][0]])]
            p=math.log(x) if x!=0 else -math.inf
            B[room]= p + prob[room]['table'][(pred[room],)]

def handle_None(sensor_data, sensor_type, key):
    
    if sensor_type=="sensor":
        # sensor_data[key]=np.random.choice(['Yes','No'], 1, p=[sensor_prob[key],1-sensor_prob[key]])[0]
        temp_room=graph_sensor[key][0]
        if sensor_prob[key]['table'][(state[temp_room], 'Yes')] > sensor_prob[key]['table'][(state[temp_room], 'No')]:
            sensor_data[key]='Yes'
        else:
            sensor_data[key]='No'
    else:
        # sensor_data[key]=np.random.choice(['Yes','No'], 1, p=[sensor_prob[key],1-sensor_prob[key]])[0]
        temp_rooms=graph_sensor[key]
        if sensor_prob[key]['table'][(state[temp_rooms[0]],state[temp_rooms[1]],  'Yes')] > sensor_prob[key]['table'][(state[temp_rooms[0]],state[temp_rooms[1]], 'No')]:
            sensor_data[key]='Yes'
        else:
            sensor_data[key]='No'
    return sensor_data

def init_B(sensor_data):
    global B
    norm_const=(3*1+0.1*(len(state.keys())-3))
    for room in graphX:
        if room in graphE:
            B[room]=rev_emission[room]['table'][(sensor_data[graphE[room][0]], state[room])]+math.log(0.1/norm_const)
        else:
            if room in ['outside', 'r12', 'r22']:
                B[room]=math.log(1/norm_const)  
            else:
                B[room]=math.log(0.1/norm_const)

def make_prediction(sensor_data, room):
    global state
    temp_prob={'dom': (room, ),
                 'table': odict([])}
    if len(graphX[room])!=0:
        for neighbour in graphX[room]:
            temp_prob['table'][('Yes', )]=temp_prob['table'].get(('Yes', ), 0)+B[room]+transition[room][neighbour]['table'][(state[room], 'Yes')]
            temp_prob['table'][('No', )]=temp_prob['table'].get(('No', ), 0)+B[room]+transition[room][neighbour]['table'][(state[room], 'No')]

    else:
        temp_prob['table'][('Yes', )]=rev_emission[room]['table'][(sensor_data[graphE[room][0]],'Yes')]
        temp_prob['table'][('No', )]=rev_emission[room]['table'][(sensor_data[graphE[room][0]],'No')]
    pred_prob[room]=temp_prob.copy()
    if temp_prob['table'][('Yes', )] > temp_prob['table'][('No', )]:
        state[room]='Yes'
    else:
        state[room]='No'

# Required dictionalies and default values
state=dict.fromkeys(graphX.keys(), 'No')
state['outside']='Yes'
state['r12']='Yes'
state['r22']='Yes'
pred_prob = {}
B = {}
def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state,B,pred_prob

    # Handle sensor data
    for key in sensor_data.keys():
        if 'reliable' in key:
            if sensor_data[key] == None:
                sensor_data=handle_None(sensor_data, 'sensor', key)
            elif sensor_data[key] == 'motion':
                sensor_data[key] = 'Yes'
            else:
                sensor_data[key] = 'No'
        if 'door' in key:
            if sensor_data[key] == None:
                sensor_data=handle_None(sensor_data, 'door', key)
            elif sensor_data[key] >  0:
                sensor_data[key] = 'Yes'
            elif sensor_data[key] == 0:
                sensor_data[key] = 'No'

    # update B     
    if len(pred_prob.keys())==0:    
        init_B(sensor_data)
    else:
        calculateB(state, pred_prob, sensor_data)

    # Based on B and sensor_data make prediction
    for room in graphX:
        make_prediction(sensor_data, room)
    
    # Handle Robot data
    robot1 = sensor_data['robot1']
    robot2 = sensor_data['robot2']
    if robot1 != None:
        r_sensor1 = robot1[1:len(robot1)-1].split(', ')
        if int(r_sensor1[1]) == 0:
            state[r_sensor1[0].strip('\'')] = 'No'
        else:
            state[r_sensor1[0].strip('\'')] = 'Yes'
    if robot2 != None:
        r_sensor2 = robot2[1:len(robot2)-1].split(', ')
        if int(r_sensor2[1]) == 0:
            state[r_sensor2[0].strip('\'')] = 'No'
        else:
            state[r_sensor2[0].strip('\'')] = 'Yes'    


    actions_dict ={}
#     if sensor_data['time']>datetime.time(17,35,0):
#         for j in range(1,36): 
#             actions_dict['lights'+str(j)] = 'off'

#     else:
    for j in range(1,36): 
        if state['r'+str(j)] == 'No' :
            actions_dict['lights'+str(j)] = 'off'
        else:
            actions_dict['lights'+str(j)] = 'on'
    return actions_dict

