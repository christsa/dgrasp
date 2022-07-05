//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir), cfgString_(cfg) {
    Yaml::Parse(cfg_, cfg);
	raisim::World::setActivationKey(raisim::Path(resourceDir + "/activation.raisim").getString());
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {
    omp_set_num_threads(cfg_["num_threads"].template As<int>());
    num_envs_ = cfg_["num_envs"].template As<int>();

    environments_.reserve(num_envs_);
    rewardInformation_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
      rewardInformation_.push_back(environments_.back()->getRewards().getStdMap());
    }

    setSeed(0);

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }
    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

    void load_object(Eigen::Ref<EigenRowMajorMatInt> &obj_idx, Eigen::Ref<EigenRowMajorMatDouble> &obj_weight, Eigen::Ref<EigenRowMajorMatDouble>& obj_dim, Eigen::Ref<EigenRowMajorMatInt>& obj_type) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->load_object(obj_idx.row(i), obj_weight.row(i), obj_dim.row(i), obj_type.row(i));
    }

  void reset_state(Eigen::Ref<EigenRowMajorMatDouble> &init_state, Eigen::Ref<EigenRowMajorMatDouble> &init_vel, Eigen::Ref<EigenRowMajorMatDouble> &obj_pose) {
#pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
          environments_[i]->reset_state(init_state.row(i), init_vel.row(i), obj_pose.row(i));
  }

    void set_goals(Eigen::Ref<EigenRowMajorMatDouble> &obj_pos, Eigen::Ref<EigenRowMajorMatDouble> &ee_pos, Eigen::Ref<EigenRowMajorMatDouble> &pose, Eigen::Ref<EigenRowMajorMatDouble> &contact_pos, Eigen::Ref<EigenRowMajorMatDouble> &vertex_normals) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_goals(obj_pos.row(i), ee_pos.row(i), pose.row(i), contact_pos.row(i), vertex_normals.row(i));
    }

  void observe(Eigen::Ref<EigenRowMajorMatDouble> &ob) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
  }

  void step(Eigen::Ref<EigenRowMajorMatDouble> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void set_root_control() {
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++)
                  environments_[i]->set_root_control();
  }


  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  const std::vector<std::map<std::string, float>>& getRewardInfo() { return rewardInformation_; }

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMatDouble> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
//    std::cout << "action i" << agentId << " " << action.row(agentId)[0] << " " <<  action.row(agentId)[1] << " " << action.row(agentId)[2] << std::endl;
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    rewardInformation_[agentId] = environments_[agentId]->getRewards().getStdMap();

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::map<std::string, float>> rewardInformation_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
