from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp_test as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.dgrasp_test import NormalSampler
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
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
parser.add_argument('-w', '--weight', type=str, default='2021-09-29-18-20-07/full_400.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-o', '--obj_id',type=int, default=1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mc','--mesh_collision', action="store_true")
parser.add_argument('-ao','--all_objects', action="store_true")
parser.add_argument('-ev','--vis_evaluate', action="store_true")
parser.add_argument('-sv','--store_video', action="store_true")
parser.add_argument('-to','--test_object_set', type=int, default=-1)
parser.add_argument('-ac','--all_contact', action="store_true")
parser.add_argument('-seed','--seed', type=int, default=1)
parser.add_argument('-itr','--num_iterations', type=int, default=3001)
parser.add_argument('-nr','--num_repeats', type=int, default=1)
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

### task specification
task_name = args.exp_name
### check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
test_inference = args.test
train_obj_id = args.obj_id
all_obj_train = True if args.all_objects else False
meta_info_dim = 4

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


num_envs = 1 if args.vis_evaluate else final_qpos.shape[0]
cfg['environment']['hand_model'] = "mano_mean_meshcoll.urdf" if args.mesh_collision else "mano_mean.urdf"
cfg['environment']['num_envs'] = 1 if args.vis_evaluate else num_envs
cfg["testing"] = True if test_inference else False
print('num envs', final_qpos.shape[0])

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

# Setting dimensions from environments
n_act = final_qpos[0].shape[0]
ob_dim = env.num_obs
act_dim = env.num_acts

### Set training step parameters
grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps
total_steps = n_steps * env.num_envs

avg_rewards = []

### Set up logging
log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name
saver = ConfigurationSaver(log_dir = log_dir,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=True)

### Set up RL algorithm
actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim-meta_info_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0,  NormalSampler(act_dim)),device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim-meta_info_dim, 1),device)

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

### Loading a pretrained model
load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor, critic, ppo.optimizer, saver.data_dir,args.cfg, store_again=False)

### Initialize the environment
env.set_goals(final_obj_pos,final_ee,final_pose,final_contact_pos,final_contacts)
env.reset_state(qpos_reset, np.zeros((num_envs,51),'float32'), obj_pose_reset)


### Evaluate trained model visually (note always the first environment gets visualized)
if args.vis_evaluate:
    ### Start recording
    if args.store_video:
        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
    for i in range(final_qpos.shape[0]):
        ### Set labels and load objects for current label (only one visualization per rollout possible)
        qpos_reset_seq = qpos_reset.copy()
        qpos_reset_seq[0] = qpos_reset_seq[i]
        final_qpos_seq = final_qpos.copy()
        final_qpos_seq[0] = final_qpos_seq[i].copy()

        obj_pose_reset_seq = obj_pose_reset.copy()
        obj_pose_reset_seq[0] = obj_pose_reset_seq[i].copy()

        final_contact_pos_seq = final_contact_pos.copy()
        final_contact_pos_seq[0] = final_contact_pos[i].copy()

        final_contacts_seq = final_contacts.copy()
        final_contacts_seq[0] = final_contacts[i].copy()

        final_obj_pos_seq = final_obj_pos.copy()
        final_ee_seq = final_ee.copy()
        final_pose_seq = final_pose.copy()


        final_obj_pos_seq[0] = final_obj_pos[i].copy()
        final_ee_seq[0] = final_ee_seq[i].copy()
        final_pose_seq[0] = final_pose_seq[i].copy()

        obj_idx_stacked[0] =  obj_idx_stacked[i].copy()
        obj_w_stacked[0] =  obj_w_stacked[i].copy()
        obj_dim_stacked[0] =  obj_dim_stacked[i].copy()
        obj_type_stacked[0] =  obj_type_stacked[i].copy()

        if i>0 and obj_idx_stacked[i-1] != obj_idx_stacked[i]:
            env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

        env.set_goals(final_obj_pos_seq,final_ee_seq,final_pose_seq,final_contact_pos_seq,final_contacts_seq)
        env.reset_state(qpos_reset_seq, np.zeros((num_envs,51)), obj_pose_reset_seq)

        set_guide=False
        obj_pose_pos_list = []
        hand_pose_list = []
        joint_pos_list = []

        time.sleep(2)
        for step in range(n_steps):
            obs = env.observe(False)

            ### Get action from policy
            action_pred = actor.architecture.architecture(torch.from_numpy(obs[:,:-meta_info_dim]).to(device))
            frame_start = time.time()

            action_ll = action_pred.cpu().detach().numpy()

            ### After grasp is established remove surface and test stability
            if step>grasp_steps:
                if not set_guide:
                    env.set_root_control()
                    set_guide=True

            reward_ll, dones = env.step(action_ll)

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

        ### Store recording
        if args.store_video:
            print('store video')
            env.stop_video_recording()
            env.turn_off_visualization()

### quantitative evaluation
else:
    disp_list, slipped_list, contact_ratio_list = [], [], []
    qpos_list, joint_pos_list, obj_pose_list = [], [], []

    set_guide=False

    for step in range(n_steps):
        obs = env.observe(False)

        action_ll = actor.architecture.architecture(torch.from_numpy(obs[:,:-4]).to(device))
        frame_start = time.time()

        ### After grasp is established remove surface and test stability
        if step>grasp_steps and not set_guide:
            obj_pos_fixed = obs[:,-4:-1].copy()
            env.set_root_control()
            set_guide=True

        ### Record slipping and displacement
        if step>(grasp_steps+1):
            slipped_list.append(obs[:,-1].copy())
            obj_disp = np.linalg.norm(obj_pos_fixed-obs[:,-4:-1],axis=-1)
            disp_list.append(obj_disp)
            obj_pos_fixed = obs[:,-4:-1].copy()

        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    ### Log quantiative results
    for obj_id in np.unique(obj_idx_stacked):
        train_obj_id = obj_id + 1

        ### compute testing window
        sim_dt = cfg['environment']['simulation_dt']
        control_dt = cfg['environment']['control_dt']
        control_steps = int(control_dt / sim_dt)
        sim_to_real_steps = 1/(control_steps * sim_dt)
        window_5s = int(5*sim_to_real_steps)

        obj_idx_array = np.where(obj_idx_stacked == obj_id)[0]

        slipped_array = np.array(slipped_list)[:].transpose()[obj_idx_array]
        disp_array = np.array(disp_list)[:].transpose()[obj_idx_array]

        slips, success_idx, disps = [], [], []
        ### evaluate slipping and sim dist
        for idx in range(slipped_array.shape[0]):
            if slipped_array[idx,:window_5s].any():
                slips.append(True)
            else:
                success_idx.append(idx)

            if slipped_array[idx].any():
                slip_step_5s =  np.clip(np.where(slipped_array[idx])[0][0]-1,1, window_5s)
                disps.append(disp_array[idx,:slip_step_5s].copy().mean())
            else:
                disps.append(disp_array[idx,:window_5s].copy().mean())

        avg_slip = 1-np.array(slips).sum()/slipped_array.shape[0]
        avg_disp =  np.array(disps).mean()*1000
        std_disp =  np.array(disps).std()*1000

        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("object: ", obj_id+1))
        print('{:<40} {:>6}'.format("success: ", '{:0.3f}'.format(avg_slip)))
        print('{:<40} {:>6}'.format("disp mean: ", '{:0.3f}'.format(avg_disp)))
        print('{:<40} {:>6}'.format("disp std: ", '{:0.3f}'.format(std_disp)))
        print('----------------------------------------------------\n')

        if not all_obj_train:
            np.save(log_dir+'/success_idxs',success_idx)


    ### Log average success rate over all objects
    if all_obj_train:
        slipped_array = np.array(slipped_list)[:].transpose()
        disp_array = np.array(disp_list)[:].transpose()

        slips, success_idx, disps = [], [], []
        ### evaluate slipping and sim dist
        for idx in range(slipped_array.shape[0]):
            if slipped_array[idx,:window_5s].any():
                slips.append(True)
            else:
                success_idx.append(idx)

            if slipped_array[idx].any():
                slip_step_5s =  np.clip(np.where(slipped_array[idx])[0][0]-1,1, window_5s)
                disps.append(disp_array[idx,:slip_step_5s].copy().mean())
            else:
                disps.append(disp_array[idx,:window_5s].copy().mean())

        avg_slip = 1-np.array(slips).sum()/slipped_array.shape[0]
        avg_disp =  np.array(disps).mean()*1000
        std_disp =  np.array(disps).std()*1000

        if len(success_idx) > 0:
            np.save(log_dir+'/success_idxs',success_idx)

        print('----------------------------------------------------')
        print('{:<40}'.format("all objects"))
        print('{:<40} {:>6}'.format("total success rate: ", '{:0.3f}'.format(avg_slip)))
        print('{:<40} {:>6}'.format("disp mean: ", '{:0.3f}'.format(avg_disp)))
        print('{:<40} {:>6}'.format("disp std: ", '{:0.3f}'.format(std_disp)))
        print('----------------------------------------------------\n')



