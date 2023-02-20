import math

from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Transformer


class Actor(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, buf_spmap_cstr, buffer_size_list, steps_per_level, problem_instance):
        super(Actor, self).__init__()

        self.transformer = Transformer(d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
                                       n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=0,
                                       n_position=100, trg_emb_prj_weight_sharing=True,
                                       scale_emb_or_prj='prj')
        self.buffer_size_list = buffer_size_list
        self.buf_spmap_cstr = buf_spmap_cstr
        self.steps_per_level = steps_per_level
        self.problem_instance = problem_instance

    def get_max_tile_fit_buffer(self, sol):
        len_dim = len('NKCPQRS')
        tile_prods = {}
        tile_prod = np.ones((len_dim,))
        steps_per_level = 7
        # sol [level*steps_per_level, 5]
        dim2note = {0: 'N', 1: 'K', 2: 'C', 3: 'P', 4: 'Q', 5: 'R', 6: 'S'}
        for level in range(1, self.num_buffer_level):
            level_sol = sol[(level - 1) * steps_per_level:level * steps_per_level, :]
            par_dims = set()
            permutation = ''
            tile_sizes = {}
            for i in range(steps_per_level):
                note = dim2note[level_sol[i, 1]]
                permutation += note
                if level_sol[i, 0] == 1:
                    par_dims.add(note)
                tile_sizes[note] = pow(2, level_sol[i, 2]) * pow(3, level_sol[i, 3]) * pow(7, level_sol[i, 4])
            print("tile_sizes: ", tile_sizes, par_dims)
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes, par_dims, timeloop_notation=False)
            print("tp_tile_sizes: ", level, level_sol, tp_tile_sizes, sp_tile_sizes)
            print(tp_tile_sizes * sp_tile_sizes, tile_sizes)
            tile_prod = (tile_prod * tp_tile_sizes * sp_tile_sizes)
            tile_prods[f'l{level}'] = tile_prod
            print(tile_prod)
        for level in range(1, self.num_buffer_level):
            input_tile, weight_tile, output_tile = self.get_input_weight_output_tile(tile_prods[f'l{level}'])
            total_tile = 0
            # total_tile += input_tile if sol[f'l{level}']['bypass']['Inputs'] is False else 0
            # total_tile += weight_tile if sol[f'l{level}']['bypass']['Weights'] is False else 0
            # total_tile += output_tile if sol[f'l{level}']['bypass']['Outputs'] is False else 0
            total_tile += input_tile
            total_tile += weight_tile
            total_tile += output_tile

    def get_remain_buffer_size(self, cur_buffer_level, trg_seq_disorder, order_action, is_cur):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = trg_seq_disorder.size(0)
        tiles = trg_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_trg_seq_disorder = copy.deepcopy(trg_seq_disorder[:, start_ind:end_ind])
            tiles *= torch.pow(2, level_trg_seq_disorder[:, :, 1]) * torch.pow(3, level_trg_seq_disorder[:, :, 2]) * \
                     torch.pow(7, level_trg_seq_disorder[:, :, 3])

        # TODO previous levels tiles
        # print(tiles.size(), tiles)
        N, K, C, P, Q, R, S = torch.unbind(tiles, dim=1)
        wstride = self.problem_instance['Wstride']
        hstride = self.problem_instance['Hstride']
        wdilation = self.problem_instance['Wdilation']
        hdilation = self.problem_instance['Hdilation']
        if cur_buffer_level == 1 or cur_buffer_level == 3:  # pe weight
            N = trg_seq_disorder.new_zeros(batch_size)
            P = trg_seq_disorder.new_zeros(batch_size)
            Q = trg_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # pe acc
            C = trg_seq_disorder.new_zeros(batch_size)
            R = trg_seq_disorder.new_zeros(batch_size)
            S = trg_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 4:  # pe input
            K = trg_seq_disorder.new_zeros(batch_size)

        # input_tile = N * (P + R - 1) * (Q + S - 1) * C
        input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C
        weight_tile = K * R * S * C
        output_tile = P * Q * K * N
        N_sub = weight_tile
        K_sub = input_tile
        C_sub = output_tile
        # P_sub = weight_tile + N * (R - 1) * (Q + S - 1) * C
        # Q_sub = weight_tile + N * (S - 1) * (P + R - 1) * C
        # R_sub = output_tile + N * (P - 1) * (Q + S - 1) * C
        # S_sub = output_tile + N * (Q - 1) * (P + R - 1) * C
        P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C
        Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C
        R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C
        S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C

        # N_coef = ((P + R - 1) * (Q + S - 1) * C + P * Q * K) * N
        # K_coef = (R * S * C + P * Q * N) * K
        # C_coef = (N * (P + R - 1) * (Q + S - 1) + K * R * S) * C
        # P_coef = (N * (Q + S - 1) * C + Q * K * N) * P
        # Q_coef = (N * (P + R - 1) * C + P * K * N) * Q
        # R_coef = (N * (Q + S - 1) * C + K * S * C) * R
        # S_coef = (N * (P + R - 1) * C + K * R * C) * S
        N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N
        K_coef = (R * S * C + P * Q * N) * K
        C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C
        P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P
        Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q
        R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R
        S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S

        if cur_buffer_level == 5:  # global buffer
            # input_tile = N * (P + R - 1) * (Q + S - 1) * C
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C
            weight_tile = trg_seq_disorder.new_zeros(batch_size)
            output_tile = P * Q * K * N
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile
            # P_sub = weight_tile + N * (R - 1) * (Q + S - 1) * C
            # Q_sub = weight_tile + N * (S - 1) * (P + R - 1) * C
            # R_sub = output_tile + N * (P - 1) * (Q + S - 1) * C
            # S_sub = output_tile + N * (Q - 1) * (P + R - 1) * C
            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C

            # N_coef = ((P + R - 1) * (Q + S - 1) * C + P * Q * K) * N
            # K_coef = P * Q * N * K
            # C_coef = (N * (P + R - 1) * (Q + S - 1)) * C
            # P_coef = (N * (Q + S - 1) * C + Q * K * N) * P
            # Q_coef = (N * (P + R - 1) * C + P * K * N) * Q
            # R_coef = (N * (Q + S - 1) * C) * R
            # S_coef = (N * (P + R - 1) * C) * S
            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N
            K_coef = P * Q * N * K
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * C
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * R
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * S
        else:
            # if is_cur:
            #     if cur_buffer_level == 1 or cur_buffer_level == 3:
            #         N_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         P_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         Q_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         N_coef = trg_seq_disorder.new_ones(batch_size)
            #         P_coef = trg_seq_disorder.new_ones(batch_size)
            #         Q_coef = trg_seq_disorder.new_ones(batch_size)
            #     elif cur_buffer_level == 2:
            #         C_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         R_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         S_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         C_coef = trg_seq_disorder.new_ones(batch_size)
            #         R_coef = trg_seq_disorder.new_ones(batch_size)
            #         S_coef = trg_seq_disorder.new_ones(batch_size)
            #     elif cur_buffer_level == 4:
            #         K_sub = trg_seq_disorder.new_ones(batch_size).fill_(buffer_size)
            #         K_coef = trg_seq_disorder.new_ones(batch_size)
            # else:
            if cur_buffer_level == 1 or cur_buffer_level == 3:
                N_sub = trg_seq_disorder.new_zeros(batch_size).float()
                P_sub = trg_seq_disorder.new_zeros(batch_size).float()
                Q_sub = trg_seq_disorder.new_zeros(batch_size).float()
                N_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:
                C_sub = trg_seq_disorder.new_zeros(batch_size).float()
                R_sub = trg_seq_disorder.new_zeros(batch_size).float()
                S_sub = trg_seq_disorder.new_zeros(batch_size).float()
                C_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                R_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                S_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 4:
                K_sub = trg_seq_disorder.new_zeros(batch_size).float()
                K_coef = trg_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([N_coef, K_coef, C_coef, P_coef, Q_coef, R_coef, S_coef], dim=1)[
            np.arange(batch_size), order_action]
        sub_arr = torch.stack([N_sub, K_sub, C_sub, P_sub, Q_sub, R_sub, S_sub], dim=1)[
            np.arange(batch_size), order_action]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        # if is_cur is False:
        #     print(coef_arr[0], sub_arr[0], remain_buffer_size[0])
        return remain_buffer_size

    def forward(self, trg_seq, trg_mask, order_mask, tile2_remain_budget, tile3_remain_budget, tile7_remain_budget,
                tile2_mask, tile3_mask, tile7_mask, parallel_mask, mode,
                cur_buffer_level, trg_seq_disorder):
        '''
        :param trg_seq    [batch, level*7, 5]
        :param trg_mask   [batch, level*7, 5]
        :param order_mask, tile2_mask, tile3_mask, tile7_mask, parallel_mask  [batch]
        :param trg_seq_disorder [batch, level*7, 3] NKCPQRS tiles
        :return: order, tile，parallel
        '''
        # print('order_mask',  order_mask)
        order_logit, tile2_logit, tile3_logit, tile7_logit, parallel_logit,\
            sp_tile2_logit, sp_tile3_logit, sp_tile7_logit = self.transformer(trg_seq)
        batch_size = trg_seq.size(0)

        # print("order_mask", trg_seq, order_logit, order_mask)
        order_score = order_logit + order_mask
        order_prob = F.softmax(order_score, dim=1)
        # print(order_logit)
        order_density = Categorical(order_prob)
        order_action = order_density.sample()
        order_log_prob = order_density.log_prob(order_action)
        order_log_prob_mask = ((order_mask == 0).sum(dim=-1) > 1).float()

        if cur_buffer_level == 4:
            if mode%self.steps_per_level == 0:
                order_action = trg_seq.new_ones(batch_size).fill_(5)
            elif mode%self.steps_per_level == 1:
                order_action = trg_seq.new_ones(batch_size).fill_(6)
            elif mode%self.steps_per_level == 2:
                order_action = trg_seq.new_ones(batch_size).fill_(0)
            elif mode%self.steps_per_level == 3:
                order_action = trg_seq.new_ones(batch_size).fill_(3)
            elif mode%self.steps_per_level == 4:
                order_action = trg_seq.new_ones(batch_size).fill_(4)
            elif mode%self.steps_per_level == 5:
                order_action = trg_seq.new_ones(batch_size).fill_(1)
            elif mode%self.steps_per_level == 6:
                order_action = trg_seq.new_ones(batch_size).fill_(2)
        else:
            if mode%self.steps_per_level == 0:
                order_action = trg_seq.new_ones(batch_size).fill_(5)
            elif mode%self.steps_per_level == 1:
                order_action = trg_seq.new_ones(batch_size).fill_(6)
            elif mode%self.steps_per_level == 2:
                order_action = trg_seq.new_ones(batch_size).fill_(0)
            elif mode%self.steps_per_level == 3:
                order_action = trg_seq.new_ones(batch_size).fill_(3)
            elif mode%self.steps_per_level == 4:
                order_action = trg_seq.new_ones(batch_size).fill_(4)
            elif mode%self.steps_per_level == 5:
                order_action = trg_seq.new_ones(batch_size).fill_(2)
            elif mode%self.steps_per_level == 6:
                order_action = trg_seq.new_ones(batch_size).fill_(1)

        tile2_log_prob = order_prob.new_zeros(batch_size)
        tile3_log_prob = order_prob.new_zeros(batch_size)
        tile7_log_prob = order_prob.new_zeros(batch_size)
        sp_tile2_log_prob = order_prob.new_zeros(batch_size)
        # sp_tile3_log_prob = order_prob.new_zeros(batch_size)
        # sp_tile7_log_prob = order_prob.new_zeros(batch_size)

        tile2_log_prob_mask = order_mask.new_zeros(batch_size)
        tile3_log_prob_mask = order_mask.new_zeros(batch_size)
        tile7_log_prob_mask = order_mask.new_zeros(batch_size)
        sp_tile2_log_prob_mask = order_mask.new_zeros(batch_size)
        # sp_tile3_log_prob_mask = order_mask.new_zeros(batch_size)
        # sp_tile7_log_prob_mask = order_mask.new_zeros(batch_size)

        # log_probs = torch.stack([order_log_prob, tile2_log_prob, tile3_log_prob, tile7_log_prob,
        #                          sp_tile2_log_prob, sp_tile3_log_prob, sp_tile7_log_prob], dim=1)
        # log_prob_masks = torch.stack([order_log_prob_mask, tile2_log_prob_mask, tile3_log_prob_mask, tile7_log_prob_mask,
        #                               sp_tile2_log_prob_mask, sp_tile3_log_prob_mask, sp_tile7_log_prob_mask], dim=1)
        log_probs = torch.stack([tile2_log_prob, tile3_log_prob, tile7_log_prob,
                                 sp_tile2_log_prob], dim=1)
        log_prob_masks = torch.stack(
            [tile2_log_prob_mask, tile3_log_prob_mask, tile7_log_prob_mask,
             sp_tile2_log_prob_mask], dim=1)

        if cur_buffer_level == len(self.buffer_size_list):
            return (order_action, None, None, None, None), log_probs, log_prob_masks

        # predict tiles
        tile2_remain_budget = tile2_remain_budget[torch.arange(0, batch_size), order_action]
        tile3_remain_budget = tile3_remain_budget[torch.arange(0, batch_size), order_action]
        tile7_remain_budget = tile7_remain_budget[torch.arange(0, batch_size), order_action]

        tile2_mask = tile2_mask[torch.arange(0, batch_size), order_action]
        tile3_mask = tile3_mask[torch.arange(0, batch_size), order_action]
        tile7_mask = tile7_mask[torch.arange(0, batch_size), order_action]

        remain_buffer_size = self.get_remain_buffer_size(cur_buffer_level, trg_seq_disorder, order_action, is_cur=True)

        for later_level in range(cur_buffer_level + 1, len(self.buffer_size_list) + 1):
            remain_buffer_size = torch.minimum(remain_buffer_size,
                                               self.get_remain_buffer_size(later_level, trg_seq_disorder,
                                                                           order_action, is_cur=False))

        tile2_max = torch.log2(torch.clamp(remain_buffer_size, min=1))
        tile2_max = torch.clamp(tile2_max.long(), min=0, max=tile2_mask.size(-1)-1)
        tile2_max = torch.minimum(tile2_max, tile2_remain_budget)

        if mode % self.steps_per_level == self.steps_per_level - 1:
            parallel_mask = parallel_mask[torch.arange(0, batch_size), order_action]
            # print(order_action, parallel_mask)
            buf_spmap_cstr = self.buf_spmap_cstr[f'l{cur_buffer_level}']
            start_ind = (cur_buffer_level - 1) * self.steps_per_level
            end_ind = cur_buffer_level * self.steps_per_level
            level_trg_seq_disorder = copy.deepcopy(trg_seq_disorder[:, start_ind:end_ind])
            used_buf_spmap = trg_seq.new_ones(batch_size)
            for i in range(self.steps_per_level):
                parallel, sp_tile2, sp_tile3, sp_tile7 = torch.unbind(level_trg_seq_disorder[:, i, 4:], dim=-1)
                used_buf_spmap *= torch.clamp(
                    parallel * torch.pow(2, sp_tile2) * torch.pow(3, sp_tile3) * torch.pow(7, sp_tile7),
                    min=1)
            remain_buf_spmap = buf_spmap_cstr / used_buf_spmap.float()

            # print("remain_buf_spmap:  ", cur_buffer_level, remain_buf_spmap)
            sp_tile2_max = torch.log2(torch.clamp(remain_buf_spmap, min=1))
            sp_tile2_max = torch.clamp(sp_tile2_max.long(), min=0, max=tile2_mask.size(-1) - 1)
            # if cur_buffer_level == 5:
            #     print(sp_tile2_max, parallel_mask[:, 1])
            sp_tile2_max = sp_tile2_max * (parallel_mask[:, 1] == 0).long()
            sp_tile2_max = torch.minimum(sp_tile2_max, tile2_max)

            sp_tile2_action = sp_tile2_max
            sp_tile2_log_prob = order_prob.new_zeros(batch_size)
            sp_tile2_log_prob_mask = order_mask.new_zeros(batch_size)

            tile2_min = sp_tile2_action

        # if mode % self.steps_per_level == self.steps_per_level - 1:
        #     print(mode, remain_buffer_size[0], tile2_max[0])
        #     tile2_action = tile2_max
        #     tile2_log_prob = order_prob.new_zeros(batch_size)
        #     tile2_log_prob_mask = order_mask.new_zeros(batch_size)
        # else:
        # tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
        # for i in range(1, tile2_mask.size(-1) + 1):
        #     # print(tile2_mask_tmp.size(), tile2_max.size())
        #     # print(cur_buffer_level, tile2_max, tile3_max, tile7_max, level_trg_seq_disorder)
        #     tile2_mask_tmp[np.arange(batch_size), tile2_max + i] = float('-inf')
        # tile2_mask = tile2_mask_tmp[:, :tile2_mask.size(-1)]


        tile2_score = tile2_logit + tile2_mask
        tile2_prob = F.softmax(tile2_score, dim=-1)
        tile2_density = Categorical(tile2_prob)
        tile2_action = tile2_density.sample()
        tile2_log_prob = tile2_density.log_prob(tile2_action)
        tile2_log_prob_mask = ((tile2_mask == 0).sum(dim=-1) > 1).float()

        remain_buffer_size = remain_buffer_size / torch.pow(2, tile2_action).float()
        tile3_max = torch.log2(torch.clamp(remain_buffer_size, min=1)) / math.log2(3)
        tile3_max = torch.clamp(tile3_max.long(), min=0, max=tile3_mask.size(-1)-1)

        # if mode % self.steps_per_level == self.steps_per_level - 1:
        #     # print(mode, remain_buf_spmap, sp_tile2_max)
        #     tile3_action = tile3_max
        #     tile3_log_prob = order_prob.new_zeros(batch_size)
        #     tile3_log_prob_mask = order_mask.new_zeros(batch_size)
        # else:
        tile3_mask_tmp = torch.cat([tile3_mask, torch.zeros_like(tile3_mask)], dim=-1)
        for i in range(1, tile3_mask.size(-1) + 1):
            tile3_mask_tmp[np.arange(batch_size), tile3_max + i] = float('-inf')
        tile3_mask = tile3_mask_tmp[:, :tile3_mask.size(-1)]

        tile3_score = tile3_logit + tile3_mask
        tile3_prob = F.softmax(tile3_score, dim=-1)
        tile3_density = Categorical(tile3_prob)
        tile3_action = tile3_density.sample()
        tile3_log_prob = tile3_density.log_prob(tile3_action)
        tile3_log_prob_mask = ((tile3_mask == 0).sum(dim=-1) > 1).float()

        remain_buffer_size = remain_buffer_size / torch.pow(3, tile3_action).float()
        tile7_max = torch.log2(torch.clamp(remain_buffer_size, min=1)) / math.log2(7)
        tile7_max = torch.clamp(tile7_max.long(), min=0, max=tile7_mask.size(-1)-1)

        # if mode % self.steps_per_level == self.steps_per_level - 1:
        #     # print(mode, remain_buf_spmap, sp_tile2_max)
        #     tile7_action = tile7_max
        #     tile7_log_prob = order_prob.new_zeros(batch_size)
        #     tile7_log_prob_mask = order_mask.new_zeros(batch_size)
        # else:
        tile7_mask_tmp = torch.cat([tile7_mask, torch.zeros_like(tile7_mask)], dim=-1)
        for i in range(1, tile7_mask.size(-1) + 1):
            tile7_mask_tmp[np.arange(batch_size), tile7_max + i] = float('-inf')
        tile7_mask = tile7_mask_tmp[:, :tile7_mask.size(-1)]
        # for i in range(batch_size):
        #     if tile2_max[i, post_order_action[i]] + 1 < tile2_mask.size(-1):
        #         tile2_mask[i, post_order_action[i], tile2_max[i, post_order_action[i]]+1:] = float('-inf')
        #     if tile3_max[i, post_order_action[i]] + 1 < tile3_mask.size(-1):
        #         tile3_mask[i, post_order_action[i], tile3_max[i, post_order_action[i]] + 1:] = float('-inf')
        #     if tile7_max[i, post_order_action[i]] + 1 < tile7_mask.size(-1):
        #         tile7_mask[i, post_order_action[i], tile7_max[i, post_order_action[i]] + 1:] = float('-inf')
        tile7_score = tile7_logit + tile7_mask
        tile7_prob = F.softmax(tile7_score, dim=-1)
        tile7_density = Categorical(tile7_prob)
        tile7_action = tile7_density.sample()
        tile7_log_prob = tile7_density.log_prob(tile7_action)
        tile7_log_prob_mask = ((tile7_mask == 0).sum(dim=-1) > 1).float()

        # print(mode, tile2_max[0], tile3_max[0], tile7_max[0])

        # predict spatial_tiles
        parallel_mask = parallel_mask[torch.arange(0, batch_size), order_action]
        # print(order_action, parallel_mask)
        buf_spmap_cstr = self.buf_spmap_cstr[f'l{cur_buffer_level}']
        start_ind = (cur_buffer_level - 1) * self.steps_per_level
        end_ind = cur_buffer_level * self.steps_per_level
        level_trg_seq_disorder = copy.deepcopy(trg_seq_disorder[:, start_ind:end_ind])
        used_buf_spmap = trg_seq.new_ones(batch_size)
        for i in range(self.steps_per_level):
            parallel, sp_tile2, sp_tile3, sp_tile7 = torch.unbind(level_trg_seq_disorder[:, i, 4:], dim=-1)
            used_buf_spmap *= torch.clamp(parallel * torch.pow(2, sp_tile2) * torch.pow(3, sp_tile3) * torch.pow(7, sp_tile7),
                                          min=1)
        remain_buf_spmap = buf_spmap_cstr / used_buf_spmap.float()

        # print("remain_buf_spmap:  ", cur_buffer_level, remain_buf_spmap)
        sp_tile2_max = torch.log2(torch.clamp(remain_buf_spmap, min=1))
        sp_tile2_max = torch.clamp(sp_tile2_max.long(), min=0, max=tile2_mask.size(-1) - 1)
        # if cur_buffer_level == 5:
        #     print(sp_tile2_max, parallel_mask[:, 1])
        sp_tile2_max = sp_tile2_max * (parallel_mask[:, 1] == 0).long()
        sp_tile2_max = torch.minimum(sp_tile2_max, tile2_action)

        if mode % self.steps_per_level == self.steps_per_level - 1:
            # print(mode, remain_buf_spmap, sp_tile2_max)
            sp_tile2_action = sp_tile2_max
            sp_tile2_log_prob = order_prob.new_zeros(batch_size)
            sp_tile2_log_prob_mask = order_mask.new_zeros(batch_size)
        else:
            sp_tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
            for i in range(1, tile2_mask.size(-1) + 1):
                sp_tile2_mask_tmp[np.arange(batch_size), sp_tile2_max + i] = float('-inf')
            sp_tile2_mask = sp_tile2_mask_tmp[:, :tile2_mask.size(-1)]

            sp_tile2_score = sp_tile2_logit + sp_tile2_mask
            sp_tile2_prob = F.softmax(sp_tile2_score, dim=-1)
            sp_tile2_density = Categorical(sp_tile2_prob)
            sp_tile2_action = sp_tile2_density.sample()
            sp_tile2_log_prob = sp_tile2_density.log_prob(sp_tile2_action)
            sp_tile2_log_prob_mask = ((sp_tile2_mask == 0).sum(dim=-1) > 1).float()

            # if cur_buffer_level == 5:
            #     print(cur_buffer_level, order_action, tile2_action, sp_tile2_mask, sp_tile2_action)

            # remain_buf_spmap = remain_buf_spmap / torch.pow(2, sp_tile2_action).float()
            # # print("remain_buf_spmap:  ", remain_buf_spmap, sp_tile2_action)
            # sp_tile3_max = torch.log2(torch.clamp(remain_buf_spmap, min=1)) / math.log2(3)
            # sp_tile3_max = torch.clamp(sp_tile3_max.long(), min=0, max=tile3_mask.size(-1) - 1)
            # sp_tile3_max = sp_tile3_max * (parallel_mask[:, 1] == 0).long()
            # sp_tile3_max = torch.minimum(sp_tile3_max, tile3_action)
            # sp_tile3_mask_tmp = torch.cat([tile3_mask, torch.zeros_like(tile3_mask)], dim=-1)
            # for i in range(1, tile3_mask.size(-1) + 1):
            #     sp_tile3_mask_tmp[np.arange(batch_size), sp_tile3_max + i] = float('-inf')
            # sp_tile3_mask = sp_tile3_mask_tmp[:, :tile3_mask.size(-1)]
            #
            # sp_tile3_score = sp_tile3_logit + sp_tile3_mask
            # sp_tile3_prob = F.softmax(sp_tile3_score, dim=-1)
            # sp_tile3_density = Categorical(sp_tile3_prob)
            # sp_tile3_action = sp_tile3_density.sample()
            # sp_tile3_log_prob = sp_tile3_density.log_prob(sp_tile3_action)

            # remain_buf_spmap = remain_buf_spmap / torch.pow(3, sp_tile3_action).float()
            # # print("remain_buf_spmap:  ", remain_buf_spmap, sp_tile3_action, "sp_tile3_action")
            # sp_tile7_max = torch.log2(torch.clamp(remain_buf_spmap, min=1)) / math.log2(7)
            # # print(sp_tile2_max, sp_tile3_max, sp_tile7_max)
            # sp_tile7_max = torch.clamp(sp_tile7_max.long(), min=0, max=tile7_mask.size(-1) - 1)
            # sp_tile7_max = sp_tile7_max * (parallel_mask[:, 1] == 0).long()
            # sp_tile7_max = torch.minimum(sp_tile7_max, tile7_action)
            # # print(sp_tile2_max, sp_tile3_max, sp_tile7_max)
            # sp_tile7_mask_tmp = torch.cat([tile7_mask, torch.zeros_like(tile7_mask)], dim=-1)
            # for i in range(1, tile7_mask.size(-1) + 1):
            #     sp_tile7_mask_tmp[np.arange(batch_size), sp_tile7_max + i] = float('-inf')
            # sp_tile7_mask = sp_tile7_mask_tmp[:, :tile7_mask.size(-1)]
            #
            # sp_tile7_score = sp_tile7_logit + sp_tile7_mask
            # sp_tile7_prob = F.softmax(sp_tile7_score, dim=-1)
            # sp_tile7_density = Categorical(sp_tile7_prob)
            # sp_tile7_action = sp_tile7_density.sample()
            # sp_tile7_log_prob = sp_tile7_density.log_prob(sp_tile7_action)

            # parallel_log_prob_mask = ((parallel_mask == 0).sum(dim=-1) > 1).float()
            # sp_tile3_log_prob_mask = ((sp_tile3_mask == 0).sum(dim=-1) > 1).float()
            # sp_tile7_log_prob_mask = ((sp_tile7_mask == 0).sum(dim=-1) > 1).float()

        sp_tile3_action = trg_seq.new_zeros(batch_size)
        sp_tile7_action = trg_seq.new_zeros(batch_size)

        parallel_action = sp_tile2_action
        parallel_action = torch.clamp(parallel_action, max=1)

        # order_log_prob_mask = ((order_mask == 0).sum(dim=-1) > 1).float()


        # log_probs = torch.stack([order_log_prob, tile2_log_prob, tile3_log_prob, tile7_log_prob,
        #                          sp_tile2_log_prob, sp_tile3_log_prob, sp_tile7_log_prob], dim=1)
        # log_prob_masks = torch.stack([order_log_prob_mask, tile2_log_prob_mask, tile3_log_prob_mask, tile7_log_prob_mask,
        #                               sp_tile2_log_prob_mask, sp_tile3_log_prob_mask, sp_tile7_log_prob_mask], dim=1)
        log_probs = torch.stack([tile2_log_prob, tile3_log_prob, tile7_log_prob,
                                 sp_tile2_log_prob], dim=1)
        log_prob_masks = torch.stack(
            [tile2_log_prob_mask, tile3_log_prob_mask, tile7_log_prob_mask,
             sp_tile2_log_prob_mask], dim=1)

        return (order_action, tile2_action, tile3_action, tile7_action,
                parallel_action, sp_tile2_action, sp_tile3_action, sp_tile7_action), log_probs, log_prob_masks
