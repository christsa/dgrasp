from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import mano_pred as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
from raisimGymTorch.helper import rotations

objects=[]
import pickle as pkl

import joblib
import open3d as o3d

MANO_TO_CONTACT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10:8,
    11:9,
    12:9,
    13:10,
    14:11,
    15:12,
    16:12,
    17:13,
    18:14,
    19:15,
    20:15,
}


IDX_TO_OBJ = {
    1: ['002_master_chef_can',0.414, 0, [0.051,0.139,0.0],[5.0000000e-01, 3.5702077e-18, 5.6900000e-01]],
    2: ['003_cracker_box', 0.453, 1, [0.06, 0.158, 0.21],[5.00000000e-01, 1.92826095e-18, 6.05000000e-01]],
    3: ['004_sugar_box', 0.514, 1, [0.038, 0.089, 0.175],[5.00000000e-01, 2.78224565e-18, 5.87000000e-01]],
    4: ['005_tomato_soup_can', 0.349, 0, [0.033, 0.101,0.0],[ 5.00000000e-01, -4.86151919e-18,  5.50000000e-01]],
    5: ['006_mustard_bottle', 0.431,2, [0.0,0.0,0.0],[5.00000000e-01, 1.94084331e-18, 5.83000000e-01]],
    6: ['007_tuna_fish_can', 0.171, 0, [0.0425, 0.033,0.0],[5.00000000e-01, 2.18616908e-18, 5.16000000e-01]],
    7: ['008_pudding_box', 0.187, 3, [0.21, 0.089, 0.035],[5.00000000e-01, 7.46278831e-19, 5.22000000e-01]],
    8: ['009_gelatin_box', 0.097, 3, [0.028, 0.085, 0.073],[ 5.00000000e-01, -4.83406252e-19,  5.18000000e-01]],
    9: ['010_potted_meat_can', 0.37, 3, [0.05, 0.097, 0.089],[ 5.00000000e-01, -2.66176702e-19,  5.52000000e-01]],
    10: ['011_banana', 0.066,2, [0.028, 0.085, 0.073],[ 5.00000000e-01, -2.66176702e-19,  5.52000000e-01]],
    11: ['019_pitcher_base', 0.178,2, [0.0,0.0,0.0],[ 0.4962, -0.0038,  0.6462]],
    12: ['021_bleach_cleanser', 0.302,2, [0.0,0.0,0.0], [ 5.00000000e-01,  3.57919433e-18,  6.10000000e-01]], # not sure about weight here
    13: ['024_bowl', 0.147,2, [0.0,0.0,0.0], [0.49,  -0.01,   0.525]],
    14: ['025_mug', 0.118,2, [0.0,0.0,0.0],[ 0.489, -0.011,  0.539]],
    15: ['035_power_drill', 0.895,2, [0.0,0.0,0.0],[5.0000000e-01, -4.2387564e-19,  5.3900000e-01]],
    16: ['036_wood_block', 0.729, 3, [0.085, 0.085, 0.2],[ 0.4867, -0.0133,  0.6167]],
    17: ['037_scissors', 0.082,2, [0.0,0.0,0.0],[0.4692, -0.0308, 0.5092]],
    18: ['040_large_marker', 0.01, 3, [0.009,0.121,0.0],[ 5.0000000e-01, -4.5294217e-18,  5.0700000e-01]],
    19: ['051_large_clamp', 0.125,2, [0.0,0.0,0.0],[ 5.00000000e-01, -4.04781164e-18,  5.08000000e-01]],
    20: ['052_extra_large_clamp', 0.102,2, [0.0,0.0,0.0],[5.00000000e-01, 4.23614509e-18, 5.17000000e-01]],
    21: ['061_foam_brick', 0.028, 1, [0.05, 0.075, 0.05],[5.00000000e-01, 2.73124056e-18, 5.25000000e-01]],
}

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
parser.add_argument('-w', '--weight', type=str, default='2021-09-29-18-20-07/full_400.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-o', '--obj_id',type=int, default=1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mi','--mean_init', action="store_true")
parser.add_argument('-seed','--seed', type=int, default=1)
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

# task specification
task_name = args.exp_name
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# config
cfg = YAML().load(open(task_path+'/cfgs/' + args.cfg, 'r'))

if args.seed != 1:
    cfg['seed']=args.seed
num_envs = cfg['environment']['num_envs']
start_offset = 0
trail_steps = 100
pre_grasp_steps = 100
trail_steps_eval = 100
trail_steps_lift = 35
no_coll_steps = 0
start_dist_thresh = 0.25
contact_threshold = 0.015
final_threshold = 0.005
reward_clip = -2.0
ob_dim_info = 157

num_repeats=cfg['environment']['num_repeats']
activations = nn.LeakyReLU
output_activation = nn.Tanh
output_act = False
inference_flag = False
all_train = False
all_obj_train = False
data_test = None
test_inference = args.test

obj_key = args.obj_id

torch.set_default_dtype(torch.double)
if not test_inference:
    data_all_obj=joblib.load("raisimGymTorch/data/dexycb_grasptta_train.pkl")
else:
    data_all_obj=joblib.load("raisimGymTorch/data/dexycb_grasptta_test.pkl")


obj_w_stacked =     np.repeat(data_all_obj[obj_key]['obj_w'],num_repeats,0)
obj_idx_stacked =   np.repeat(data_all_obj[obj_key]['obj_idx'],num_repeats,0)
obj_dim_stacked =   np.repeat(data_all_obj[obj_key]['obj_dim'],num_repeats,0)
obj_type_stacked =  np.repeat(data_all_obj[obj_key]['obj_type'],num_repeats,0)

num_envs = data_all_obj[obj_key]['final_qpos'].shape[0]

final_qpos =  np.repeat(data_all_obj[obj_key]['final_qpos'],num_repeats,0)
final_obj_pos = np.repeat(data_all_obj[obj_key]['final_obj'],num_repeats,0)
final_pose = np.repeat(data_all_obj[obj_key]['final_qpos'],num_repeats,0)
final_ee = np.repeat(data_all_obj[obj_key]['final_wpos'],num_repeats,0)
final_contact_pos = np.repeat(data_all_obj[obj_key]['final_wpos'],num_repeats,0)
final_vertex_normals = np.repeat(data_all_obj[obj_key]['final_contacts'],num_repeats,0)

x_offset = IDX_TO_OBJ[obj_key][4][0]
y_offset = IDX_TO_OBJ[obj_key][4][1]
z_offset = IDX_TO_OBJ[obj_key][4][2]


final_qpos[:,:3] += np.array(IDX_TO_OBJ[obj_key][4])
final_obj_pos [:,:3] += np.array(IDX_TO_OBJ[obj_key][4])

final_ee= final_ee.reshape(-1,21,3)
final_ee[...,:3] += np.array(IDX_TO_OBJ[obj_key][4])
final_ee = final_ee.reshape(-1,63)
#cfg['environment']['obj']=IDX_TO_OBJ[data['ycb_objects'][data['object_id']]][0]+"/textured_meshlab.obj"
# create environment from the configuration file

cfg['environment']['hand_model'] = "mano_mean.urdf"
#num_envs = final_qpos.shape[0]*num_repeats # if not test_inference else len(data_one_obj_test.keys())
cfg['environment']['num_envs'] = 1 if inference_flag else num_envs*num_repeats
print('num envs', num_envs*num_repeats)

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# obj_idx_array = np.expand_dims(np.repeat(obj_idx_stacked,1),-1).astype('int32')
# obj_weight_array = np.expand_dims(np.repeat(obj_w_stacked,1),-1)

env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

n_act = final_qpos[0].shape[0]



ob_dim = env.num_obs
act_dim = env.num_acts

# Training
trail_steps_add = trail_steps_eval #if inference_flag else trail_steps
grasp_steps = pre_grasp_steps #if inference_flag else  qpos_stacked[0].shape[0]
trail_steps_lift_add = trail_steps_lift #if inference_flag else 0

n_steps = grasp_steps + trail_steps_lift_add + trail_steps_add #math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], output_activation, activations, ob_dim-ob_dim_info, act_dim, output_act),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)


critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], output_activation, activations, ob_dim-ob_dim_info, 1, output_act),
                           device)


saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path  + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"])

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=num_envs*num_repeats,
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False
              )


if mode == 'retrain':
    load_param(saver.data_dir.split('2023')[0]+weight_path, env, actor, critic, ppo.optimizer, saver.data_dir, args.cfg)


env.set_goals(final_obj_pos,final_ee,final_pose,final_contact_pos,final_vertex_normals)
env.reset_state(final_qpos, np.zeros((num_envs*num_repeats,51),'float64'), final_obj_pos)


eval_dict = {}
start = time.time()

reward_ll_sum = 0
done_sum = 0
average_dones = 0.



fpos_ftips = []

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)


obj_pose_reset = final_obj_pos.copy()
qpos_reset = final_qpos.copy()
qpos_reset[:,:3] -= (obj_pose_reset[:,:3]-final_qpos[:,:3])
if args.mean_init:
    qpos_reset[:,6:] = np.zeros_like(qpos_reset[:,6:])

for update in range(3001):
    start = time.time()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if mode == "retrain":

        while True:
            fpos_ftips = []
            #env.turn_on_visualization()
            #env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            for i in range(num_envs):

                qpos_reset_seq = qpos_reset.copy()
                qpos_reset_seq[0] = qpos_reset_seq[i*num_repeats]
                final_qpos_seq = final_qpos.copy()
                final_qpos_seq[0] = final_qpos_seq[i*num_repeats].copy()

                obj_pose_reset_seq = obj_pose_reset.copy()
                obj_pose_reset_seq[0] = obj_pose_reset_seq[i*num_repeats].copy()

                final_contact_pos_seq = final_contact_pos.copy()
                final_contact_pos_seq[0] = final_contact_pos[i*num_repeats].copy()

                final_vertex_normals_seq = final_vertex_normals.copy()
                final_vertex_normals_seq[0] = final_vertex_normals[i*num_repeats].copy()

                final_obj_pos_seq = final_obj_pos.copy()
                final_ee_seq = final_ee.copy()
                final_pose_seq = final_pose.copy()


                final_obj_pos_seq[0] = final_obj_pos[i*num_repeats].copy()
                final_ee_seq[0] = final_ee_seq[i*num_repeats].copy()
                final_pose_seq[0] = final_pose_seq[i*num_repeats].copy()

                obj_idx_stacked[0] =  obj_idx_stacked[i*num_repeats].copy()
                obj_w_stacked[0] =  obj_w_stacked[i*num_repeats].copy()
                obj_dim_stacked[0] =  obj_dim_stacked[i*num_repeats].copy()
                obj_type_stacked[0] =  obj_type_stacked[i*num_repeats].copy()

             
                env.set_goals(final_obj_pos_seq,final_ee_seq,final_pose_seq,final_contact_pos_seq,final_vertex_normals_seq)
                env.reset_state(qpos_reset_seq, np.zeros((num_envs*num_repeats,51),'float64'), obj_pose_reset_seq)

                dd=True
                set_guide=False

                for step in range(n_steps):
                    obs = env.observe(False)

                    action_ll = actor.architecture.architecture(torch.from_numpy(obs.astype('float64')).to(device))#loaded_graph.architecture(torch.from_numpy(obs.astype('float32')).cpu())
                    frame_start = time.time()

                    if step>grasp_steps and dd:
                        if not set_guide:
                            env.set_rootguidance()
                            set_guide=True

                    if step>(grasp_steps+trail_steps_lift_add):
                        #measure displacement
                        pass

                    reward_ll, dones = env.step(action_ll.cpu().detach().numpy().astype('float64')) # np.tile(qpos_gc,(num_envs,1)).astype('float64')) # #action_ll.cpu().detach().numpy()) #  # qpos_gc.astype('float64'))#

                    print('step ', step)
                    print(reward_ll)
                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

    if update % cfg['environment']['eval_every_n'] == 0:# and update > 500:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        env.turn_on_visualization()
 
        env.save_scaling(saver.data_dir, str(update))

    random_noise_pos = np.random.uniform([-0.02, -0.02, 0.01],[0.02, 0.02, 0.01], (num_envs*num_repeats,3)).copy()
    random_noise_qpos = np.random.uniform(-0.05,0.05, (num_envs*num_repeats,48)).copy()
    qpos_noisy_reset = qpos_reset.copy()
    qpos_noisy_reset[:,:3] += random_noise_pos[:,:3]
    qpos_noisy_reset[:,3:] += random_noise_qpos[:,:]
    env.set_goals(final_obj_pos,final_ee,final_pose,final_contact_pos,final_vertex_normals)
    env.reset_state(qpos_noisy_reset, np.zeros((num_envs*num_repeats,51),'float64'), obj_pose_reset)

    for step in range(n_steps):
        obs = env.observe().astype('float64')

        # if np.isnan(obs).any():
        #     print('early')
        action = ppo.observe(obs)
        #



        reward, dones = env.step(action.astype('float64'))#.cpu().detach().numpy().astype('float64'))
        reward.clip(min=reward_clip)

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)
        #print(reward)
    # take st step to get value obs
    #print("obsy" , obs)

    obs = env.observe().astype('float64')
    # if (probs==0).any():
    #     obs[:] = obs[resample_idxs]

    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()

    mean_file_name = saver.data_dir + "/rewards.txt"
    np.savetxt(mean_file_name, avg_rewards)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

    if np.isnan(np.exp(actor.distribution.std.cpu().detach().numpy())).any():
        print('resetting env')
        env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
