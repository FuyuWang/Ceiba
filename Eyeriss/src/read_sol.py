import os
import json
import pickle
import numpy as np
import yaml

# rl_dir = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Simba/report/simba_arch_4x'
# rl_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Simba/report/simba_arch_4x_pe/sampled_episodes_32'
# cosa_dir = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/simba_v3_4x_pe'
# rl_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Eyeriss/report/eyeriss_2x/sampled_episodes_32'
# cosa_dir = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/eyeriss_v3_2x'

dnn = 'resnet50'
# dnn = 'vgg16'
input_sizes = [1, 2, 4, 8, 16]
layers = yaml.load(open('../in_config/{}_problems/layers.yaml'.format(dnn), 'r'), Loader=yaml.SafeLoader)
resnet50_layer_ids = [0, 1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 43, 44, 45, 46, 47, 48]
# vgg16_layer_ids = [0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14]
# layer_ids = [0, 2, 12, 13, 14, 24, 27, 47]
num_layers = len(resnet50_layer_ids)
total_rl_fitnesses = np.zeros((num_layers, len(input_sizes)))
# total_rl_8_fitnesses = np.zeros((num_layers, len(input_sizes)))
# total_rl_16_fitnesses = np.zeros((num_layers, len(input_sizes)))
# total_rl_small_32_fitnesses = np.zeros((num_layers, len(input_sizes)))
total_rl_fitnesses2 = np.zeros((num_layers, len(input_sizes)))
# total_rl_fitnesses3 = np.zeros((num_layers, len(input_sizes)))
total_cosa_fitnesses = np.zeros((num_layers, len(input_sizes)))
# for i in range(num_layers-1):
for i, layer_id in enumerate(resnet50_layer_ids):
    rl_fitnesses = []
    cosa_fitnesses = []

    rl_8_fitnesses = []
    rl_16_fitnesses = []
    rl_small_32_fitnesses = []
    rl_fitnesses2 = []
    rl_fitnesses3 = []

    for input_size in input_sizes:
        # rl_chkpt = pickle.load(open(os.path.join(rl_dir, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'), 'rb'))
        # rl_fitness = -1 * rl_chkpt['best_fitness']
        # # rl_fitness = -1 * rl_chkpt['best_energy']*1e6
        # # rl_fitness = rl_chkpt['best_energy']*1e6*rl_chkpt['best_fitness']
        # rl_fitnesses.append(rl_fitness)
        try:
            cosa_chkpt = json.load(open(os.path.join(cosa_dir, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'tc.dict.json'), 'r'))
            cosa_fitness =  list(cosa_chkpt.values())[0]['cycle']
            # cosa_fitness =  list(cosa_chkpt.values())[0]['energy']
            # cosa_fitness =  list(cosa_chkpt.values())[0]['energy'] * list(cosa_chkpt.values())[0]['cycle']
        except:
            cosa_fitness = float('inf')
        cosa_fitnesses.append(cosa_fitness)

        rl_chkpt2 = pickle.load(open(
            os.path.join(rl_dir2, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'),
            'rb'))
        rl_fitness2 = -1 * rl_chkpt2['best_fitness_record'][9]
        # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][19]
        # rl_fitness2 = -1 * rl_chkpt2['best_fitness_record'][19]*1e6
        rl_fitnesses2.append(rl_fitness2)

        # rl_chkpt3 = pickle.load(open(
        #     os.path.join(rl_dir3, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'),
        #     'rb'))
        # rl_fitness3 = -1 * rl_chkpt3['best_fitness']
        # rl_fitnesses3.append(rl_fitness3)

    # print(i, layer_id)
    # print('cosa: ', cosa_fitnesses)
    # print('rl: ', rl_fitnesses)
    # print('rl2: ', rl_fitnesses2)
    # print('rl3: ', rl_fitnesses3)
    # print('rl_8: ', rl_8_fitnesses)
    # print('rl_16: ', rl_16_fitnesses)
    # print('rl_small_32: ', rl_small_32_fitnesses)

    # total_rl_fitnesses[i] = rl_fitnesses
    total_rl_fitnesses2[i] = rl_fitnesses2
    # total_rl_fitnesses3[i] = rl_fitnesses3
    total_cosa_fitnesses[i] = cosa_fitnesses
    # print(rl_fitnesses, cosa_fitnesses)
# print(total_rl_fitnesses.shape, total_cosa_fitnesses.shape)
# total_rl_fitnesses[total_cosa_fitnesses==float('inf')] = 0
total_rl_fitnesses2[total_cosa_fitnesses==float('inf')] = 0
# total_rl_fitnesses3[total_cosa_fitnesses==float('inf')] = 0
total_cosa_fitnesses[total_cosa_fitnesses==float('inf')] = 0
# print(total_cosa_fitnesses[:-1].mean(axis=0) / total_rl_fitnesses[:-1].mean(axis=0))
print(total_cosa_fitnesses.mean(axis=0) / total_rl_fitnesses2.mean(axis=0))
# print(total_cosa_fitnesses[:-1].mean(axis=0) / total_rl_fitnesses3[:-1].mean(axis=0))
# np.save('./npy/total_rl_fitnesses_simba_arch_4x.npy', total_rl_fitnesses2)
# np.save('./npy/total_cosa_fitnesses_simba_arch_4x.npy', total_cosa_fitnesses)
