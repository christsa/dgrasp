from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import mano_gen as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.env.envs.mano_gen.mano_pca as pca
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

TEST_OBJS = {
    0: [3,4,5],
    1: [21, 9, 20],
    2: [1, 12,14],
    3: [2, 6, 10],
    4: [1, 16, 20],
    5: [7, 11, 15],
    6: [4, 17, 18],
    7: [3,12,17],
    8: [3, 5, 17],
    9: [2, 5, 15],
    10: []
}

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
parser.add_argument('-w', '--weight', type=str, default='2021-09-29-18-20-07/full_400.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-pr','--prior', action="store_true")
parser.add_argument('-o', '--obj_id',type=int, default=1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mc','--mesh_collision', action="store_true")
parser.add_argument('-ao','--all_objects', action="store_true")
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
pre_grasp_steps = 60
trail_steps_eval = 100
trail_steps_lift = 35
no_coll_steps = 0
start_dist_thresh = 0.25
contact_threshold = 0.015
final_threshold = 0.005
reward_clip = -2.0

num_repeats= args.num_repeats # cfg['environment']['num_repeats']
if args.prior:
    cfg['environment']['prior'] = True
activations = nn.LeakyReLU
output_activation = nn.Tanh
output_act = False
inference_flag = False

if args.all_objects:
    all_obj_train = True
    if args.test_object_set != -1:
        test_obj_set = TEST_OBJS[args.test_object_set]
    else:
        test_obj_set = []
else:
    all_obj_train = False
    test_obj_set = []

all_train = False

data_test = None
test_inference = args.test

train_obj_id = args.obj_id

torch.set_default_dtype(torch.double)

if all_train:
    data_all=joblib.load("raisimGymTorch/data/seq_collection_all_subjects_complete.pkl")
else:
    data_all=joblib.load("raisimGymTorch/data/dexycb_train_corrected.pkl")
    if args.test or all_obj_train:
        data_test=joblib.load("raisimGymTorch/data/dexycb_test_corrected.pkl")

data_one_obj = {}
data_one_obj_test = {}
data_train_split = {}
data_test_split = {}
counter = 1
test_freq = 4
for subj in data_all.keys():
    data_test_split[subj] = {}
    data_train_split[subj] = {}

    if all_obj_train and args.test_object_set != -1:
        for keys in data_all[subj].keys():
            data_one_obj[keys] = data_all[subj][keys]
        for keys in data_test[subj].keys():
            data_one_obj[keys] = data_test[subj][keys]

    elif data_test is not None and test_inference:

        for keys in data_test[subj].keys():
            if not all_obj_train:
                if data_test[subj][keys]['ycb_objects'][data_test[subj][keys]['object_id']] == train_obj_id:
                    data_one_obj[keys] = data_test[subj][keys]
            else:
                data_one_obj[keys] = data_test[subj][keys]

    # if counter % test_freq == 0:
    else:
        for keys in data_all[subj].keys():
            if not all_obj_train:
                if data_all[subj][keys]['ycb_objects'][data_all[subj][keys]['object_id']] == train_obj_id:
                    data_one_obj[keys] = data_all[subj][keys]
                    data_one_obj[keys]['subj'] = subj
            else:
                data_one_obj[keys] = data_all[subj][keys]



        #     data_test_split[subj][keys] = data_all[subj][keys]
        # else:
        #     data_train_split[subj][keys]  = data_all[subj][keys]

        counter += 1


# joblib.dump(data_train_split, 'data_train_split.pkl')
# joblib.dump(data_test_split, 'data_test_split.pkl')

qpos_list =[]
obj_pose_list, obj_pose_zero_list = [],[]
obj_idx_list = []
obj_w_list, obj_pose_reset_list, qpos_reset_list, obj_dim_list, obj_type_list = [], [], [], [], []
final_qpos_list, final_obj_pos_list, final_pose_list,final_ee_list, final_contact_pos_list, final_vertex_normals_list = [], [], [], [], [], []
final_ee_rel_list = []
frame_start_list = []
seq_id_list = []
subject_list = []


for key in data_one_obj.keys():
    data_seq = data_one_obj[key]

    num_obj = len(data_seq['ycb_objects'])
    obj_id = data_seq['object_id']

    if args.test:
        if data_seq["ycb_objects"][data_seq['object_id']] not in test_obj_set:
            continue
    else:
        if data_seq["ycb_objects"][data_seq['object_id']] in test_obj_set:
            continue

    found_start_pos = False
    start_dist_thresh_temp = start_dist_thresh
    while not found_start_pos:
        start_offset_all = np.where(np.linalg.norm(data_seq['qpos'][:,:3].copy()-data_seq["ycb_pose"].reshape(-1, num_obj, 7)[:,obj_id,:3].copy(),axis=-1)<=start_dist_thresh_temp)[0]

        if start_offset_all.shape[0] !=0:
            start_offset = start_offset_all[0]
            found_start_pos = True
        else:
            start_dist_thresh_temp += 0.001


    seq_id_list.append(key)
    #subject_list.append(data_seq['subj'])
    frame_start_list.append(start_offset)

    found_grasp_pos = False
    final_threshold_temp = final_threshold
    while not found_grasp_pos:
        final_pos_t =  np.where(np.linalg.norm(data_seq["ycb_pose"].reshape(-1, num_obj, 7)[start_offset+1:,obj_id,:3]-data_seq["ycb_pose"].reshape(-1, num_obj, 7)[start_offset:-1,obj_id,:3],axis=-1)>final_threshold_temp)[0]
        if final_pos_t.shape[0] !=0:
            final_pos = final_pos_t[0]
            found_grasp_pos = True
        else:
            final_threshold_temp -= 0.001



    qpos = data_seq['qpos'][start_offset:].copy()
    qpos[...,2] += 0.5
    qpos_list.append(qpos)

    obj_pose = data_seq["ycb_pose"].copy().reshape(-1, num_obj, 7)[start_offset:,obj_id].copy() #[10:, 0]
    obj_pose[:,2] += 0.5
    obj_pose_zero = data_seq["ycb_pose"].reshape(-1, num_obj, 7)[:,obj_id].copy() #[10:, 0]
    obj_pose_zero[:,2] += 0.5
    obj_pose_list.append(obj_pose)
    obj_w_list.append(IDX_TO_OBJ[data_seq["ycb_objects"][data_seq['object_id']]][1])
    obj_idx_list.append(data_seq["ycb_objects"][data_seq['object_id']]-1)
    obj_type_list.append(IDX_TO_OBJ[data_seq["ycb_objects"][data_seq['object_id']]][2])
    obj_dim_list.append(np.array(IDX_TO_OBJ[data_seq["ycb_objects"][data_seq['object_id']]][3]))

    mesh_03d = o3d.io.read_triangle_mesh(home_path + "/rsc/meshes_simplified/"+IDX_TO_OBJ[data_seq["ycb_objects"][data_seq['object_id']]][0]+"/textured_simple.obj", enable_post_processing=True)
    mesh_03d.remove_duplicated_vertices()

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.009, height=0.121, resolution=10, split=1)
    #o3d.io.write_triangle_mesh('cylinder.obj', cylinder)
    #o3d.visualization.draw_geometries([cylinder])

    rel_ftip_pos = data_seq["ftip_rpos"][start_offset:].reshape(qpos.shape[0],-1,3)#[:,:1]
    final_relftip_pos = np.tile(rel_ftip_pos[final_pos],(np.asarray(mesh_03d.vertices).shape[0],1,1))
    verts = np.expand_dims(np.asarray(mesh_03d.vertices)[:,:3],axis=1)
    diff_vert_fpos = np.linalg.norm(final_relftip_pos-verts,axis=-1)

    min_vert_idx = np.argmin(diff_vert_fpos,axis=0)
    min_vert_dist = np.min(diff_vert_fpos,axis=0)

    min_vertices = verts[min_vert_idx,0].flatten()

    mesh_03d.compute_vertex_normals()
    normals=np.asarray(mesh_03d.vertex_normals)
    min_normals=normals[min_vert_idx].flatten()

    #for i in range(min_vert_dist.shape[0]):
    idx_below_thresh = np.where(min_vert_dist<contact_threshold)[0]
    target_idxs = [MANO_TO_CONTACT[idx] for idx in idx_below_thresh]

    if args.all_contact:
        target_contacts = np.ones(16)
    else:
        target_contacts = np.zeros(16)
        target_contacts[target_idxs]=1
        target_contacts[-1]=1

    ftip_pos = data_seq["ftip_wpos"][start_offset:].reshape(qpos.shape[0],-1,3)

    final_qpos_list.append(qpos[final_pos].copy())
    final_obj_pos_list.append(obj_pose[final_pos].copy())
    final_pose_list.append(qpos[final_pos,3:].copy())
    final_ee_list.append(ftip_pos.reshape(qpos.shape[0],-1)[final_pos])
    final_ee_rel_list.append(rel_ftip_pos.reshape(qpos.shape[0],-1)[final_pos])
    final_contact_pos_list.append(min_vertices)
    final_vertex_normals_list.append(target_contacts)

    reset_pos = 0#np.max([0,final_pos-10])#0#final_pos - 10 #1 # np.max([1,final_pos-10])
    qpos_reset_list.append(qpos[reset_pos])
    obj_pose_reset_list.append(obj_pose_zero[0])


qpos_stacked = np.repeat(np.array(qpos_list),num_repeats,0)
obj_pose_stacked = np.repeat(np.array(obj_pose_list),num_repeats,0)
obj_w_stacked = np.repeat(np.array(obj_w_list),num_repeats,0).astype('float64')
obj_idx_stacked = np.repeat(np.array(obj_idx_list),num_repeats,0).astype('int32')

obj_dim_stacked = np.repeat(np.vstack(obj_dim_list),num_repeats,0).astype('float64')
obj_type_stacked = np.repeat(np.array(obj_type_list),num_repeats,0).astype('int32')


qpos_reset =  np.repeat(np.vstack(qpos_reset_list),num_repeats,0).astype('float64')
obj_pose_reset =  np.repeat(np.vstack(obj_pose_reset_list),num_repeats,0).astype('float64')

final_qpos = np.repeat(np.vstack(final_qpos_list),num_repeats,0).astype('float64')
final_obj_pos = np.repeat(np.vstack(final_obj_pos_list),num_repeats,0).astype('float64')
final_pose = np.repeat(np.vstack(final_pose_list),num_repeats,0).astype('float64')
final_ee = np.repeat(np.vstack(final_ee_list),num_repeats,0).astype('float64')
final_ee_rel =  np.repeat(np.vstack(final_ee_rel_list),num_repeats,0).astype('float64')
final_contact_pos = np.repeat(np.vstack(final_contact_pos_list),num_repeats,0).astype('float64')
final_vertex_normals = np.repeat(np.vstack(final_vertex_normals_list),num_repeats,0).astype('float64')

dict = {}



# rel_wpos = final_qpos[:,:3]-final_obj_pos[:,:3]
# rotmats = np.array([rotations.quat2mat(quat) for quat in final_obj_pos[:,3:]])
# rotmats_hand = np.array([rotations.euler2mat(quat) for quat in final_qpos[:,3:6]])
# rotmats_hand_transl = final_qpos[:,:3].copy()
# rotmats_hand_transl -= final_obj_pos[:,:3]
# # rotmats_hand_transl -= np.array([0.0957, 0.0064, 0.0062])
# #final_qpos[:,:3] -= final_obj_pos[:,:3]
#
#
# rotmats_hand_pose = np.matmul(rotmats.swapaxes(1,2),rotmats_hand)
# final_qpos[:,3:6] = np.array([rotations.mat2euler(rotmat) for rotmat in rotmats_hand_pose])
#
#
# final_qpos[:,:3] = np.matmul(rotmats.swapaxes(1,2),np.expand_dims(rotmats_hand_transl,-1))[:,:,0]
#
# dict['qpos'] = final_qpos
# dict['obj_pos'] = final_obj_pos
# dict['obj_idx'] = obj_idx_stacked
# joblib.dump(dict,'data_grasp_labels_train.pkl')



#cfg['environment']['obj']=IDX_TO_OBJ[data['ycb_objects'][data['object_id']]][0]+"/textured_meshlab.obj"
# create environment from the configuration file

cfg['environment']['hand_model'] = "mano_mean_meshcoll.urdf" if args.mesh_collision else "mano_mean.urdf"
num_envs = len(qpos_reset_list) # if not test_inference else len(data_one_obj_test.keys())
cfg['environment']['num_envs'] = 1 if inference_flag else num_envs*num_repeats
print('num envs', num_envs*num_repeats)
cfg["testing"] = True if test_inference else False

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# obj_idx_array = np.expand_dims(np.repeat(obj_idx_stacked,1),-1).astype('int32')
# obj_weight_array = np.expand_dims(np.repeat(obj_w_stacked,1),-1)

env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

n_act = final_qpos[0].shape[0]



ob_dim = env.num_obs
act_dim = 12 if args.prior else env.num_acts

# Training
trail_steps_add = trail_steps_eval #if inference_flag else trail_steps
grasp_steps = pre_grasp_steps #if inference_flag else  qpos_stacked[0].shape[0]
trail_steps_lift_add = trail_steps_lift #if inference_flag else 0

n_steps = grasp_steps + trail_steps_lift_add + trail_steps_add #math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

#
actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], output_activation, activations, ob_dim, act_dim, output_act),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)


critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], output_activation, activations, ob_dim, 1, output_act),
                           device)

if mode == 'retrain':
    test_dir=True
else:
    test_dir=False

saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=test_dir)
# tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

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
    load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor, critic, ppo.optimizer, saver.data_dir,args.cfg)


#qpos_reset[:,6:]=0
env.set_goals(final_obj_pos,final_ee,final_pose,final_contact_pos,final_vertex_normals)
env.reset_state(qpos_reset, np.zeros((num_envs*num_repeats,51),'float64'), obj_pose_reset)




for update in range(args.num_iterations):
    start = time.time()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if mode == "retrain":

        while True:
            fpos_ftips = []
            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
            data_seq = {}
            for i in range(num_envs):
                data_seq[i] = {}
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

                env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
                env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)
                env.set_goals(final_obj_pos_seq,final_ee_seq,final_pose_seq,final_contact_pos_seq,final_vertex_normals_seq)
                env.reset_state(qpos_reset_seq, np.zeros((num_envs*num_repeats,51),'float64'), obj_pose_reset_seq)
                dd=True
                set_guide=False
                obj_pose_pos_list = []
                hand_pose_list = []
                joint_pos_list = []
                import pdb
                pdb.set_trace()
                for step in range(n_steps):
                    obs = env.observe(False)

                    action_pred = actor.architecture.architecture(torch.from_numpy(obs.astype('float64')).to(device))#loaded_graph.architecture(torch.from_numpy(obs.astype('float32')).cpu())
                    #action_pred[:,6:]=torch.clip(action_pred[:,6:], -1.5, 1.5)
                    frame_start = time.time()

                    pose_before = torch.Tensor(obs[:,6:51]).to(device)

                    if args.prior:
                        pca_pose = pca.mano_pca(action_pred[:,6:], mano_path=home_path+"/raisimGymTorch/models/MANO_RIGHT.pkl", device=device)
                        pca_pose_diff = pca_pose -pose_before

                        action_ll = torch.cat((action_pred[:,:6],pca_pose_diff),-1)
                    else:
                        action_ll = action_pred

                    if step>grasp_steps and dd:

                        if not set_guide:

                            env.set_rootguidance()
                            set_guide=True


                    if step>(grasp_steps+trail_steps_lift_add):
                        #measure displacement
                        pass

                    reward_ll, dones = env.step(action_ll.cpu().detach().numpy().astype('float64')) # np.tile(qpos_gc,(num_envs,1)).astype('float64')) # #action_ll.cpu().detach().numpy()) #  # qpos_gc.astype('float64'))#
                    #import pdb
                    #pdb.set_trace()
                    #print('step ', step)
                    print(reward_ll)
                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)


            #stop_record = False

            #if stop_record:
            print('store video')
            env.stop_video_recording()
            env.turn_off_visualization()

    #
    # reward_ll, dones = env.step(np.expand_dims(action,0))


    if update % cfg['environment']['eval_every_n'] == 0 and update > 500:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
    
        env.save_scaling(saver.data_dir, str(update))
  
    def euler_noise_to_quat(quats, palm_pose, ang_noise_in):
        angle_noise = ang_noise_in

        noises = np.random.uniform(-angle_noise,angle_noise, (num_envs*num_repeats,3))
        eulers_palm_mats = np.array([rotations.euler2mat(pose) for pose in palm_pose]).copy()
        eulers_mats =  np.array([rotations.quat2mat(quat) for quat in quats])

        rotmats_list = np.array([rotations.euler2mat(noise) for noise in noises])

        eulers_new = np.matmul(rotmats_list,eulers_mats)
        eulers_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_new])

        eulers_palm_new = np.matmul(rotmats_list,eulers_palm_mats)
        eulers_palm_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_palm_new])

        quat_list = [rotations.euler2quat(noise) for noise in eulers_rotmated]

        return np.array(quat_list), eulers_new, eulers_palm_rotmated

    random_noise = np.random.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 0.3], (num_envs*num_repeats,3)).copy()
    pose_noise = 0.5
    final_obj_pos_random = final_obj_pos.copy()
    final_obj_pos_random[:,:3] += random_noise[:,:3]
    perturbed_obj_pose, rotmats, eulers_palm_new = euler_noise_to_quat(final_obj_pos_random[:,3:].copy(), final_pose[:,:3].copy(), pose_noise)

    final_obj_pos_random[:,3:] = perturbed_obj_pose

    final_pose_perturbed = final_pose.copy()
    final_pose_perturbed[:,:3] = eulers_palm_new


    rotmats_neutral = np.tile(np.eye(3),(num_envs,1,1))

    final_ee_random_or = np.squeeze(np.matmul(np.expand_dims(rotmats,1),np.expand_dims(final_ee_rel.copy().reshape(num_envs*num_repeats,-1,3),-1)),-1)
    final_ee_random_or += np.expand_dims(final_obj_pos_random[:,:3],1)

    random_noise_pos = np.random.uniform([-0.02, -0.02, 0.01],[0.02, 0.02, 0.01], (num_envs*num_repeats,3)).copy()
    random_noise_qpos = np.random.uniform(-0.05,0.05, (num_envs*num_repeats,48)).copy()
    qpos_noisy_reset = qpos_reset.copy()
    qpos_noisy_reset[:,:3] += random_noise_pos[:,:3]
    qpos_noisy_reset[:,3:] += random_noise_qpos[:,:]

    #dict_success = joblib.load('raisimGymTorch/data/dict_success.pkl')

    env.set_goals(final_obj_pos_random,final_ee_random_or.reshape(num_envs*num_repeats,-1),final_pose_perturbed,final_contact_pos,final_vertex_normals)
    env.reset_state(qpos_noisy_reset, np.zeros((num_envs*num_repeats,51),'float64'), obj_pose_reset)

    for step in range(n_steps):
        obs = env.observe().astype('float64')

        pose_before = torch.Tensor(obs[:,6:51]).to(device)
        action_pred = ppo.observe(obs)

        if args.prior:
            #action_pred[:,6:]=np.clip(action_pred[:,6:], -1.5, 1.5)

            pca_pose = pca.mano_pca(torch.Tensor(action_pred[:,6:]).to(device), mano_path=home_path+"/raisimGymTorch/models/MANO_RIGHT.pkl", device=device)
            pca_pose_diff = pca_pose-pose_before

            action = np.concatenate((action_pred[:,:6],pca_pose_diff.cpu().detach().numpy()),-1)
        else:
            action = action_pred


        reward, dones = env.step(action.astype('float64'))#.cpu().detach().numpy().astype('float64'))
        reward.clip(min=reward_clip)
        # if dones.any():

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)
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
