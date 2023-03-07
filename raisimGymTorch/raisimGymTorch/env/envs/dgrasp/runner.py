from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.dgrasp import NormalSampler
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
import joblib

IDX_TO_OBJ = {
    1: ['002_master_chef_can',0.414, 0, [0.051,0.139,0.0]],
    2: ['003_cracker_box', 0.453, 1, [0.06, 0.158, 0.21]],
    3: ['004_sugar_box', 0.514, 1, [0.038, 0.089, 0.175]],
    4: ['005_tomato_soup_can', 0.349, 0, [0.033, 0.101,0.0]],
    5: ['006_mustard_bottle', 0.431,2, [0.0,0.0,0.0]],
    6: ['007_tuna_fish_can', 0.171, 0, [0.0425, 0.033,0.0]],
    7: ['008_pudding_box', 0.187, 3, [0.21, 0.089, 0.035]],
    8: ['009_gelatin_box', 0.097, 3, [0.028, 0.085, 0.073]],
    9: ['010_potted_meat_can', 0.37, 3, [0.05, 0.097, 0.089]],
    10: ['011_banana', 0.066,2, [0.028, 0.085, 0.073]],
    11: ['019_pitcher_base', 0.178,2, [0.0,0.0,0.0]],
    12: ['021_bleach_cleanser', 0.302,2, [0.0,0.0,0.0]], # not sure about weight here
    13: ['024_bowl', 0.147,2, [0.0,0.0,0.0]],
    14: ['025_mug', 0.118,2, [0.0,0.0,0.0]],
    15: ['035_power_drill', 0.895,2, [0.0,0.0,0.0]],
    16: ['036_wood_block', 0.729, 3, [0.085, 0.085, 0.2]],
    17: ['037_scissors', 0.082,2, [0.0,0.0,0.0]],
    18: ['040_large_marker', 0.01, 3, [0.009,0.121,0.0]],
    19: ['051_large_clamp', 0.125,2, [0.0,0.0,0.0]],
    20: ['052_extra_large_clamp', 0.102,2, [0.0,0.0,0.0]],
    21: ['061_foam_brick', 0.028, 1, [0.05, 0.075, 0.05]],
}

### configuration of command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
parser.add_argument('-w', '--weight', type=str, default='full_400.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-pr','--prior', action="store_true")
parser.add_argument('-o', '--obj_id',type=int, default=1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mc','--mesh_collision', action="store_true")
parser.add_argument('-ao','--all_objects', action="store_true")
parser.add_argument('-ev','--evaluate', action="store_true")
parser.add_argument('-to','--test_object_set', type=int, default=-1)
parser.add_argument('-ac','--all_contact', action="store_true")
parser.add_argument('-seed','--seed', type=int, default=1)
parser.add_argument('-itr','--num_iterations', type=int, default=3001)
parser.add_argument('-nr','--num_repeats', type=int, default=10)
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

### task specification
task_name = args.exp_name
### check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.set_default_dtype(torch.double)
### directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

### load config
cfg = YAML().load(open(task_path+'/cfgs/' + args.cfg, 'r'))

### set seed
if args.seed != 1:
    cfg['seed']=args.seed

### get experiment parameters
num_envs = cfg['environment']['num_envs']
pre_grasp_steps = cfg['environment']['pre_grasp_steps']
trail_steps = cfg['environment']['trail_steps']
reward_clip = cfg['environment']['reward_clip']
test_inference = args.test
train_obj_id = args.obj_id
all_obj_train = True if args.all_objects else False

### get network parameters
num_repeats= args.num_repeats
activations = nn.LeakyReLU
output_activation = nn.Tanh

### Load data labels
if not args.test:
    dict_labels=joblib.load("raisimGymTorch/data/dexycb_train_labels.pkl")
else:
    dict_labels=joblib.load("raisimGymTorch/data/dexycb_test_labels.pkl")

### Load data labels for all objects
if all_obj_train:
    obj_w_list, obj_pose_reset_list, qpos_reset_list, obj_dim_list, obj_type_list, obj_idx_list_list = [], [], [], [], [], []
    final_qpos_list, final_obj_pos_list, final_pose_list,final_ee_list, final_ee_rel_list, final_contact_pos_list, final_contacts_list = [], [], [], [], [], [], []
    ### Iterate through all objects and add to single dict
    for obj_key in dict_labels.keys():
        final_qpos_list.append(dict_labels[obj_key]['final_qpos'])
        final_obj_pos_list.append(dict_labels[obj_key]['final_obj_pos'])
        final_pose_list.append(dict_labels[obj_key]['final_pose'])
        final_ee_list.append(dict_labels[obj_key]['final_ee'])
        final_ee_rel_list.append(dict_labels[obj_key]['final_ee_rel'])
        final_contact_pos_list.append(dict_labels[obj_key]['final_contact_pos'])
        final_contacts_list.append(dict_labels[obj_key]['final_contacts'])

        obj_w_list.append(dict_labels[obj_key]['obj_w_stacked'])
        obj_dim_list.append(dict_labels[obj_key]['obj_dim_stacked'])
        obj_type_list.append(dict_labels[obj_key]['obj_type_stacked'])
        obj_idx_list_list.append(dict_labels[obj_key]['obj_idx_stacked'])
        obj_pose_reset_list.append(dict_labels[obj_key]['obj_pose_reset'])
        qpos_reset_list.append(dict_labels[obj_key]['qpos_reset'])

    final_qpos = np.repeat(np.vstack(final_qpos_list),num_repeats,0).astype('float32')
    final_obj_pos = np.repeat(np.vstack(final_obj_pos_list),num_repeats,0).astype('float32')
    final_pose = np.repeat(np.vstack(final_pose_list),num_repeats,0).astype('float32')
    final_ee = np.repeat(np.vstack(final_ee_list),num_repeats,0).astype('float32')
    final_ee_rel =  np.repeat(np.vstack(final_ee_rel_list),num_repeats,0).astype('float32')
    final_contact_pos = np.repeat(np.vstack(final_contact_pos_list),num_repeats,0).astype('float32')
    final_contacts = np.repeat(np.vstack(final_contacts_list),num_repeats,0).astype('float32')

    obj_w_stacked = np.repeat(np.hstack(obj_w_list),num_repeats,0).astype('float32')
    obj_dim_stacked = np.repeat(np.vstack(obj_dim_list),num_repeats,0).astype('float32')
    obj_type_stacked = np.repeat(np.hstack(obj_type_list),num_repeats,0)
    obj_idx_stacked = np.repeat(np.hstack(obj_idx_list_list),num_repeats,0)
    obj_pose_reset = np.repeat(np.vstack(obj_pose_reset_list),num_repeats,0).astype('float32')
    qpos_reset = np.repeat(np.vstack(qpos_reset_list),num_repeats,0).astype('float32')

### Load labels for single object
else:
    final_qpos = np.repeat(dict_labels[train_obj_id]['final_qpos'],num_repeats,0).astype('float32')
    final_obj_pos = np.repeat(dict_labels[train_obj_id]['final_obj_pos'],num_repeats,0).astype('float32')
    final_pose = np.repeat(dict_labels[train_obj_id]['final_pose'],num_repeats,0).astype('float32')
    final_ee = np.repeat(dict_labels[train_obj_id]['final_ee'],num_repeats,0).astype('float32')
    final_ee_rel = np.repeat(dict_labels[train_obj_id]['final_ee_rel'],num_repeats,0).astype('float32')
    final_contact_pos = np.repeat(dict_labels[train_obj_id]['final_contact_pos'],num_repeats,0).astype('float32')
    final_contacts = np.repeat(dict_labels[train_obj_id]['final_contacts'],num_repeats,0).astype('float32')

    obj_w_stacked = np.repeat(dict_labels[train_obj_id]['obj_w_stacked'],num_repeats,0).astype('float32')
    obj_dim_stacked = np.repeat(dict_labels[train_obj_id]['obj_dim_stacked'],num_repeats,0).astype('float32')
    obj_type_stacked = np.repeat(dict_labels[train_obj_id]['obj_type_stacked'],num_repeats,0)
    obj_idx_stacked = np.repeat(dict_labels[train_obj_id]['obj_idx_stacked'],num_repeats,0)
    obj_pose_reset = np.repeat(dict_labels[train_obj_id]['obj_pose_reset'],num_repeats,0).astype('float32')
    qpos_reset = np.repeat(dict_labels[train_obj_id]['qpos_reset'],num_repeats,0).astype('float32')


num_envs = final_qpos.shape[0]
cfg['environment']['hand_model'] = "mano_mean_meshcoll.urdf" if args.mesh_collision else "mano_mean.urdf"
cfg['environment']['num_envs'] = 1 if args.evaluate else num_envs
cfg["testing"] = True if test_inference else False
print('num envs', num_envs)

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

### Setting dimensions from environments
n_act = final_qpos[0].shape[0]
ob_dim = env.num_obs
act_dim = env.num_acts

### Set training step parameters
grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps
total_steps = n_steps * env.num_envs

if mode == 'retrain':
    test_dir=True
else:
    test_dir=False

### Set up logging
saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=test_dir)
#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update


### Set up RL algorithm
actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)),device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim, 1),device)


ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=num_envs,
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False
              )

### If loading a pretrained model
if mode == 'retrain' or args.evaluate:
    load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor, critic, ppo.optimizer, saver.data_dir,args.cfg)


### Initialize the environment
env.set_goals(final_obj_pos,final_ee,final_pose,final_contact_pos,final_contacts)
env.reset_state(qpos_reset, np.zeros((num_envs,51),'float32'), obj_pose_reset)

avg_rewards = []
for update in range(args.num_iterations):
    start = time.time()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    ### Store policy
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        env.save_scaling(saver.data_dir, str(update))

    ### Add some noise to initial hand position
    random_noise_pos = np.random.uniform([-0.02, -0.02, 0.01],[0.02, 0.02, 0.01], (num_envs,3)).copy()
    random_noise_qpos = np.random.uniform(-0.05,0.05, (num_envs,48)).copy()
    qpos_noisy_reset = qpos_reset.copy()
    qpos_noisy_reset[:,:3] += random_noise_pos[:,:3]
    qpos_noisy_reset[:,3:] += random_noise_qpos[:,:]

    ### Run episode rollouts
    env.reset_state(qpos_noisy_reset, np.zeros((num_envs,51),'float32'), obj_pose_reset)
    for step in range(n_steps):
        obs = env.observe().astype('float32')
        action = ppo.act(obs)
        reward, dones = env.step(action.astype('float32'))
        reward.clip(min=reward_clip)
        ppo.step(value_obs=obs, rews=reward, dones=dones)

        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)
    obs = env.observe().astype('float32')

    ### Update policy
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()

    ### Log results
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

