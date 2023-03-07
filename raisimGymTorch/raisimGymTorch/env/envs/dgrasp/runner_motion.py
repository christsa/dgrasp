from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.dgrasp import NormalSampler
import os
import os.path
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import joblib
from raisimGymTorch.helper import utils

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
parser.add_argument('-pr','--prior', action="store_true")
parser.add_argument('-o', '--obj_id',type=int, default=-1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mc','--mesh_collision', action="store_true")
parser.add_argument('-ao','--all_objects', action="store_true")
parser.add_argument('-ev','--evaluate', action="store_true")
parser.add_argument('-sv','--store_video', action="store_true")
parser.add_argument('-to','--test_object_set', type=int, default=-1)
parser.add_argument('-ac','--all_contact', action="store_true")
parser.add_argument('-seed','--seed', type=int, default=1)
parser.add_argument('-itr','--num_iterations', type=int, default=3001)
parser.add_argument('-nr','--num_repeats', type=int, default=1)
parser.add_argument('--motion_synthesis_extra_steps', type=int, default=50)

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
test_inference = args.test
train_obj_id = args.obj_id
all_obj_train = True if args.all_objects else False

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

    final_qpos = np.vstack(final_qpos_list).astype('float32')
    final_obj_pos = np.vstack(final_obj_pos_list).astype('float32')
    final_pose = np.vstack(final_pose_list).astype('float32')
    final_ee = np.vstack(final_ee_list).astype('float32')
    final_ee_rel =  np.vstack(final_ee_rel_list).astype('float32')
    final_contact_pos = np.vstack(final_contact_pos_list).astype('float32')
    final_contacts = np.vstack(final_contacts_list).astype('float32')

    obj_w_stacked = np.hstack(obj_w_list).astype('float32')
    obj_dim_stacked = np.vstack(obj_dim_list).astype('float32')
    obj_type_stacked = np.hstack(obj_type_list)
    obj_idx_stacked = np.hstack(obj_idx_list_list)
    obj_pose_reset = np.vstack(obj_pose_reset_list).astype('float32')
    qpos_reset = np.vstack(qpos_reset_list).astype('float32')
### Load labels for single object
else:
    final_qpos = np.array(dict_labels[train_obj_id]['final_qpos']).astype('float32')
    final_obj_pos = np.array(dict_labels[train_obj_id]['final_obj_pos']).astype('float32')
    final_pose = np.array(dict_labels[train_obj_id]['final_pose']).astype('float32')
    final_ee = np.array(dict_labels[train_obj_id]['final_ee']).astype('float32')
    final_ee_rel = np.array(dict_labels[train_obj_id]['final_ee_rel']).astype('float32')
    final_contact_pos = np.array(dict_labels[train_obj_id]['final_contact_pos']).astype('float32')
    final_contacts = np.array(dict_labels[train_obj_id]['final_contacts']).astype('float32')

    obj_w_stacked = np.array(dict_labels[train_obj_id]['obj_w_stacked']).astype('float32')
    obj_dim_stacked = np.array(dict_labels[train_obj_id]['obj_dim_stacked']).astype('float32')
    obj_type_stacked = np.array(dict_labels[train_obj_id]['obj_type_stacked'])
    obj_idx_stacked = np.array(dict_labels[train_obj_id]['obj_idx_stacked'])
    obj_pose_reset = np.array(dict_labels[train_obj_id]['obj_pose_reset']).astype('float32')
    qpos_reset = np.array(dict_labels[train_obj_id]['qpos_reset']).astype('float32')


num_envs = 1 
cfg['environment']['hand_model'] = "mano_mean_meshcoll.urdf" if args.mesh_collision else "mano_mean.urdf"
cfg['environment']['num_envs'] = num_envs
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
n_steps = grasp_steps  + trail_steps + args.motion_synthesis_extra_steps
total_steps = n_steps * env.num_envs

avg_rewards = []

log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name
saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=True)

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

### Loading a pretrained model
load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor, critic, ppo.optimizer, saver.data_dir,args.cfg, store_again=False)
fname = log_dir+'/success_idxs.npy'
success_idxs = np.load(fname) if(os.path.isfile(fname)) else np.arange(final_qpos.shape[0])

if all_obj_train and train_obj_id != -1:
    obj_success_idx = np.where(obj_idx_stacked[success_idxs] == train_obj_id-1)[0]
    success_idxs=success_idxs[obj_success_idx] if obj_success_idx.shape[0] > 0 else np.where(obj_idx_stacked == train_obj_id-1)[0]


### Load dictionary of object target 6D poses (displacements to the initial position)
eval_dict = joblib.load('raisimGymTorch/data/motion_eval_dict_easy.pkl')
obj_target_pos = eval_dict['target_pos']
obj_target_ang_noise = eval_dict['ang_noise']

### Initialize the environment
env.set_goals(final_obj_pos,final_ee,final_pose,final_contact_pos,final_contacts)
env.reset_state(qpos_reset, np.zeros((num_envs,51),'float32'), obj_pose_reset)

### Evaluate trained model visually (note always the first environment gets visualized)


last_idx = -1
cc = 0

while True:
    i = success_idxs[np.random.randint(success_idxs.shape[0])]

    rand_6D_idx = np.random.randint(obj_target_pos.shape[0])
    ### Set labels and load objects for current episode and target 6D pose
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
    final_ee_rel_seq = final_ee_rel.copy()

    final_obj_pos_seq[0] = final_obj_pos[i].copy()
    final_ee_seq[0] = final_ee_seq[i].copy()
    final_ee_rel_seq[0] = final_ee_rel_seq[i].copy()
    final_pose_seq[0] = final_pose_seq[i].copy()

    obj_idx_stacked[0] =  obj_idx_stacked[i].copy()
    obj_w_stacked[0] =  obj_w_stacked[i].copy()
    obj_dim_stacked[0] =  obj_dim_stacked[i].copy()
    obj_type_stacked[0] =  obj_type_stacked[i].copy()


    ### Add position displacement to object position
    final_obj_pos_random = final_obj_pos_seq.copy()
    final_obj_pos_random[:,:3] += obj_target_pos[rand_6D_idx]

    ### Add angle displacement to object pose
    ang_noise =  np.repeat(obj_target_ang_noise[rand_6D_idx].reshape(1,3),final_obj_pos_random.shape[0],0)
    perturbed_obj_pose, rotmats, eulers_palm_new = utils.euler_noise_to_quat(final_obj_pos_random[:,3:].copy(), final_pose_seq[:,:3].copy(), ang_noise)

    final_obj_pos_random[:,3:] = perturbed_obj_pose
    final_pose_perturbed = final_pose_seq.copy()
    final_pose_perturbed[:,:3] = eulers_palm_new

    rotmats_neutral = np.tile(np.eye(3),(num_envs,1,1))

    ### Convert position goal features to the new, distorted object pose
    final_ee_random_or = np.squeeze(np.matmul(np.expand_dims(rotmats,1),np.expand_dims(final_ee_rel_seq.copy().reshape(num_envs,-1,3),-1)),-1)
    final_ee_random_or += np.expand_dims(final_obj_pos_random[:,:3],1)

    ### Reload env if new object
    if last_idx==-1 or obj_idx_stacked[last_idx]!=obj_idx_stacked[i]:
        env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

    last_idx = i

    env.set_goals(final_obj_pos_random,final_ee_random_or.reshape(num_envs,-1).astype('float32'),final_pose_perturbed,final_contact_pos_seq,final_contacts_seq)
    env.reset_state(qpos_reset_seq, np.zeros((num_envs,51),'float32'), obj_pose_reset_seq)

    time.sleep(2)

    set_guide=False
    obj_pose_pos_list = []
    hand_pose_list = []
    joint_pos_list = []

    ### Start recording
    if args.store_video:
        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_"+task_name+'.mp4')

    for step in range(n_steps):
        obs = env.observe(False)

        ### Get action from policy
        action_pred = actor.architecture.architecture(torch.from_numpy(obs.astype('float32')).to(device))
        frame_start = time.time()

        action_ll = action_pred.cpu().detach().numpy()

        ### After grasp is established (set to motion synthesis mode)
        if step>grasp_steps:
            if not set_guide:
                env.set_root_control()
                set_guide=True


        reward_ll, dones = env.step(action_ll.astype('float32'))

        ### early exit if object not picked up successfully
        if np.linalg.norm(obs[0,213:216]) > 0.4 or dones[0]:
            break

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)
    cc += 1

    ### Store recording
    if args.store_video:
        env.stop_video_recording()
        env.turn_off_visualization()
        print('stored video')



