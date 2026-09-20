// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "../../RaisimGymEnv.hpp"
#include "raisim/math.hpp"

namespace raisim {

/// Physical right-hand GRAB tracking. The mug is only placed during reset.
/// Lengths are metres, angles radians, velocities m/s and rad/s, impulses N*s.
class ENVIRONMENT : public RaisimGymEnv {
 public:
  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable)
      : RaisimGymEnv(resourceDir, cfg), random_(0) {
    world_ = std::make_unique<raisim::World>();
    world_->addGround();
    world_->setERP(cfg["contact_erp"].As<double>(30.0));
    world_->setDefaultMaterial(0.8, 0.0, 0.0);
    set_friction(0.8);
    table_height_ = cfg["table_height"].As<double>(0.5);
    table_dimensions_ << cfg["table_length"].As<double>(2.0), cfg["table_width"].As<double>(1.0),
                         cfg["table_thickness"].As<double>(0.5);
    table_pose_.setZero(7);
    table_pose_.head(3) << 1.25, 0.0, table_height_ - 0.5 * table_dimensions_[2];
    table_pose_[3] = 1.0;
    load_set_ = cfg["load_set"].As<std::string>("grab_mug");
    domain_randomization_ = cfg["domain_randomization"].As<bool>(false);
    std::string hand_model = cfg["hand_model_r"].As<std::string>("rhand_mano_low_mass.urdf");
    const bool table_collision = cfg["enable_table_collision"].As<bool>(true);
    const CollisionGroup table_mask = table_collision ? COLLISION(1) : 0;
    hand_ = world_->addArticulatedSystem(
        resourceDir + "/mano_double/" + hand_model, "", {}, COLLISION(0),
        table_mask | COLLISION(2) | COLLISION(63));
    hand_->setName("grab_right_hand");
    if (hand_->getGeneralizedCoordinateDim() != 51 || hand_->getDOF() != 51)
      throw std::runtime_error("GRAB tracking requires the original 51-DoF MANO hand URDF");
    hand_->setControlMode(ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    set_gains(1.0);
    hand_->setGeneralizedForce(Eigen::VectorXd::Zero(51));
    hand_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(51));
    table_ = world_->addBox(table_dimensions_[0], table_dimensions_[1], table_dimensions_[2], 1.0,
                           "table", COLLISION(1),
                           (table_collision ? COLLISION(0) : 0) | COLLISION(2));
    // RaiSim 1.1.6 switches a body's collision group to 63 when made static.
    // Its explicit mask must therefore gate hand contacts for the regression toggle.
    table_->setBodyType(BodyType::STATIC);
    table_->setPosition(table_pose_[0], table_pose_[1], table_pose_[2]);
    table_->setAppearance("0.45 0.31 0.2 1");

    limits_ = hand_->getJointLimits();
    for (size_t i = 0; i < contact_bodies_.size(); ++i)
      contact_mapping_[hand_->getBodyIdx(contact_bodies_[i])] = i;
    actionDim_ = 51;
    gsDim_ = 199;
    obDim_r_ = 383;
    obDim_l_ = 1;
    state_.setZero(gsDim_);
    observation_.setZero(obDim_r_);
    reference_hand_.setZero(51);
    reference_object_.setZero(7);
    reference_object_[3] = 1.0;
    reference_joints_.setZero(63);
    reference_velocity_.setZero(51);
    previous_action_.setZero(51);
    initial_hand_.setZero(51);
    initial_hand_velocity_.setZero(51);
    initial_object_.setZero(13);
    initial_object_[3] = 1.0;
    base_origin_.setZero();
    reset_base_origin_.setZero();
    Yaml::Node reward_configuration;
    std::string reward_yaml;
    for (const std::string name : {"wrist_position", "wrist_rotation", "finger_pose", "joint_position",
                                  "object_position", "object_rotation", "contact", "action",
                                  "smoothness", "table_penetration", "object_penetration"})
      reward_yaml += name + ":\n  coeff: 1.0\n";
    Yaml::Parse(reward_configuration, reward_yaml);
    rewards_r_.initializeFromConfigurationFile(reward_configuration);
    if (visualizable && cfg["visualize"].As<bool>(false)) {
      server_ = std::make_unique<RaisimServer>(world_.get());
      server_->launchServer();
    }
  }

  void init() final {}
  void setSeed(int seed) final { random_.seed(seed); }
  void set_rootguidance() final {}

  /// Position the configured static tabletop before resetting an episode.
  /// Dimensions are xyz [m]; pose is world xyz [m] plus unit quaternion wxyz.
  void add_stage(const Eigen::Ref<EigenVec>& dimensions, const Eigen::Ref<EigenVec>& pose) final {
    if (dimensions.size() != 3 || pose.size() != 7 || !dimensions.allFinite() || !pose.allFinite())
      throw std::runtime_error("add_stage requires finite table dimensions3 and pose7");
    if ((dimensions.cast<double>() - table_dimensions_).cwiseAbs().maxCoeff() > 1e-4)
      throw std::runtime_error("Configure table_length, table_width, table_thickness to match reference table dimensions");
    table_pose_ = pose.cast<double>();
    if (table_pose_.segment(3, 4).norm() < 1e-8) throw std::runtime_error("The table quaternion is zero");
    table_pose_.segment(3, 4).normalize();
    table_->setPosition(table_pose_[0], table_pose_[1], table_pose_[2]);
    table_->setOrientation(table_pose_[3], table_pose_[4], table_pose_[5], table_pose_[6]);
    update_observation();
  }

  void load_articulated(const std::string& object_model) final {
    if (object_) throw std::runtime_error("Create a fresh GRAB environment to change its mug asset");
    const std::string object_path = !object_model.empty() && object_model.front() == '/'
        ? object_model : resourceDir_ + "/" + load_set_ + "/" + object_model;
    object_ = world_->addArticulatedSystem(
        object_path, "", {}, COLLISION(2),
        COLLISION(0) | COLLISION(1) | COLLISION(63));
    object_->setName("grab_mug");
    if (object_->getGeneralizedCoordinateDim() != 7 || object_->getDOF() != 6)
      throw std::runtime_error("The native GRAB mug must be a free rigid URDF with 7 coordinates and 6 velocities");
    object_->setControlMode(ControlMode::FORCE_AND_TORQUE);
    object_->setGeneralizedForce(Eigen::VectorXd::Zero(6));
    nominal_masses_ = object_->getMass();
    nominal_inertias_ = object_->getInertia();
    Eigen::VectorXd pose = Eigen::VectorXd::Zero(7);
    pose[2] = table_height_ + 0.2;
    pose[3] = 1.0;
    object_->setState(pose, Eigen::VectorXd::Zero(6));
  }

  void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                   const Eigen::Ref<EigenVec>& init_state_l,
                   const Eigen::Ref<EigenVec>& init_vel_r,
                   const Eigen::Ref<EigenVec>& init_vel_l,
                   const Eigen::Ref<EigenVec>& obj_pose) final {
    if (!object_ || (init_state_r.size() != 51 && init_state_r.size() != 54) || init_vel_r.size() != 51 ||
        (obj_pose.size() != 7 && obj_pose.size() != 13))
      throw std::runtime_error("reset_state requires hand51 or hand/base54, handvelocity51, and mug7 or mug13 arrays");
    initial_hand_ = init_state_r.head(51).cast<double>();
    reset_base_origin_ = initial_hand_.head(3);
    if (init_state_r.size() == 54) reset_base_origin_ = init_state_r.tail(3).cast<double>();
    // GRAB Euler sequences may be unwrapped across multiple turns. Joint
    // limits are finite, so initialize the equivalent principal-angle pose.
    for (int i = 3; i < 51; ++i) initial_hand_[i] = wrap(initial_hand_[i]);
    initial_hand_velocity_ = init_vel_r.cast<double>();
    initial_object_.setZero();
    initial_object_.head(7) = obj_pose.head(7).cast<double>();
    if (obj_pose.size() == 13) initial_object_.tail(6) = obj_pose.tail(6).cast<double>();
    if (!initial_hand_.allFinite() || !initial_hand_velocity_.allFinite() || !initial_object_.allFinite() ||
        !reset_base_origin_.allFinite())
      throw std::runtime_error("reset_state received nonfinite data");
    double quaternion_norm = initial_object_.segment(3, 4).norm();
    if (quaternion_norm < 1e-8) throw std::runtime_error("The mug reset quaternion is zero");
    initial_object_.segment(3, 4) /= quaternion_norm;
    has_initial_state_ = true;
    reset();
  }

  void reset() final {
    if (!object_ || !has_initial_state_) return;
    base_origin_ = reset_base_origin_;
    hand_->setBasePos_e(base_origin_);
    hand_->setBaseOrientation_e(Eigen::Matrix3d::Identity());
    Eigen::VectorXd hand_q = initial_hand_;
    hand_q.head(3) -= base_origin_;
    hand_->setState(hand_q, initial_hand_velocity_);
    hand_->setPdTarget(hand_q, Eigen::VectorXd::Zero(51));
    hand_->setGeneralizedForce(Eigen::VectorXd::Zero(51));
    object_->setState(initial_object_.head(7), initial_object_.tail(6));
    object_->setGeneralizedForce(Eigen::VectorXd::Zero(6));
    double mass_scale = 1.0, gains_scale = 1.0, friction = 0.8;
    if (domain_randomization_) {
      mass_scale = std::uniform_real_distribution<double>(0.8, 1.2)(random_);
      gains_scale = std::uniform_real_distribution<double>(0.9, 1.1)(random_);
      friction = std::uniform_real_distribution<double>(0.6, 1.0)(random_);
    }
    for (size_t i = 0; i < nominal_masses_.size(); ++i) {
      object_->getMass()[i] = nominal_masses_[i] * mass_scale;
      object_->getInertia()[i].e() = nominal_inertias_[i].e() * mass_scale;
    }
    object_->updateMassInfo();
    set_gains(gains_scale);
    set_friction(friction);
    reference_hand_ = initial_hand_;
    reference_object_ = initial_object_.head(7);
    reference_velocity_ = initial_hand_velocity_;
    previous_action_.setZero();
    phase_ = 0.0;
    expected_contact_ = 0.0;
    rewards_r_.reset();
    update_state();
    state_.segment(178, 21).setZero();  // Previous-episode contacts are stale until the first integration.
    reference_joints_ = state_.segment(115, 63);
    update_observation();
  }

  /// obj_pos_r: mug xyz + quaternion wxyz [m, unitless], reward only.
  /// ee_pos_r: 21 world joint xyz values [m].
  /// pose_r: world wrist xyz, intrinsic XYZ Euler root, then 45 Euler finger angles [m, rad].
  /// qpos_r: phase/contact2, optionally followed by desired generalized velocity51 [m/s, rad/s].
  void set_goals_r(const Eigen::Ref<EigenVec>& obj_pos_r,
                   const Eigen::Ref<EigenVec>& ee_pos_r,
                   const Eigen::Ref<EigenVec>& pose_r,
                   const Eigen::Ref<EigenVec>& qpos_r) final {
    if (obj_pos_r.size() != 7 || ee_pos_r.size() != 63 || pose_r.size() != 51 ||
        (qpos_r.size() != 2 && qpos_r.size() != 53))
      throw std::runtime_error("set_goals_r requires mug7, joints63, hand51, phase/contact2 or phase/contact/velocity53");
    reference_object_ = obj_pos_r.cast<double>();
    reference_joints_ = ee_pos_r.cast<double>();
    reference_hand_ = pose_r.cast<double>();
    if (!reference_object_.allFinite() || !reference_joints_.allFinite() ||
        !reference_hand_.allFinite() || !qpos_r.allFinite() || reference_object_.segment(3, 4).norm() < 1e-8)
      throw std::runtime_error("Invalid tracking reference");
    reference_object_.segment(3, 4).normalize();
    phase_ = std::clamp(static_cast<double>(qpos_r[0]), 0.0, 1.0);
    expected_contact_ = std::clamp(static_cast<double>(qpos_r[1]), 0.0, 1.0);
    reference_velocity_.setZero();
    if (qpos_r.size() == 53) {
      reference_velocity_ = qpos_r.tail(51).cast<double>();
      reference_velocity_.head(3) = reference_velocity_.head(3).cwiseMax(-5.0).cwiseMin(5.0);
      reference_velocity_.tail(48) = reference_velocity_.tail(48).cwiseMax(-20.0).cwiseMin(20.0);
    }
    update_observation();
  }

  // The original ten-argument GraspXL affordance API is deliberately unsupported.
  void set_goals(const Eigen::Ref<EigenVec>&, const Eigen::Ref<EigenVec>&,
                 const Eigen::Ref<EigenVec>&, const Eigen::Ref<EigenVec>&,
                 const Eigen::Ref<EigenVec>&, const Eigen::Ref<EigenVec>&,
                 const Eigen::Ref<EigenVec>&, const Eigen::Ref<EigenVec>&,
                 const Eigen::Ref<EigenVec>&, const Eigen::Ref<EigenVec>&) final {
    throw std::runtime_error("Use set_goals_r with explicit GRAB motion references");
  }

  float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    if (!has_initial_state_ || action_r.size() != 51 || !action_r.allFinite())
      throw std::runtime_error("Invalid tracking action or missing episode reset");
    Eigen::VectorXd action = action_r.cast<double>().cwiseMax(-1.0).cwiseMin(1.0);
    Eigen::VectorXd target = reference_hand_;
    target.head(3) -= base_origin_;
    target.head(3) += 0.03 * action.head(3);
    target.segment(3, 3) += 0.20 * action.segment(3, 3);
    target.tail(45) += 0.20 * action.tail(45);
    auto current_q = hand_->getGeneralizedCoordinate().e();
    for (int i = 3; i < 51; ++i)
      target[i] = current_q[i] + wrap(target[i] - current_q[i]);
    Eigen::VectorXd target_velocity = reference_velocity_;
    for (int i = 0; i < 51; ++i) {
      if ((target[i] <= limits_[i][0] && target_velocity[i] < 0.0) ||
          (target[i] >= limits_[i][1] && target_velocity[i] > 0.0)) target_velocity[i] = 0.0;
      target[i] = std::clamp(target[i], limits_[i][0], limits_[i][1]);
    }
    hand_->setPdTarget(target, target_velocity);
    // Feed-forward compensation supports only the hand's own weight. The mug
    // has zero applied forces and is moved solely by gravity and contact.
    Eigen::VectorXd force = Eigen::VectorXd::Zero(51);
    force[2] = hand_->getTotalMass() * 9.81;
    hand_->setGeneralizedForce(force);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); ++i) {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    update_state();
    if (!state_.allFinite()) {
      reward_sum_[0] = -10.0f;
      reward_sum_[1] = 0.0f;
      return reward_sum_.data();
    }
    const double wrist_error = (state_.head(3) - reference_hand_.head(3)).squaredNorm();
    const double wrist_angle = rotation_angle(state_.segment(3, 3), reference_hand_.segment(3, 3));
    double finger_error = 0.0;
    for (int i = 6; i < 51; ++i) finger_error += std::pow(wrap(state_[i] - reference_hand_[i]), 2);
    finger_error /= 45.0;
    const double joint_error = (state_.segment(115, 63) - reference_joints_).squaredNorm() / 21.0;
    const double object_error = (state_.segment(102, 3) - reference_object_.head(3)).squaredNorm();
    const double object_dot = std::abs(state_.segment(105, 4).dot(reference_object_.segment(3, 4)));
    const double object_angle = 2.0 * std::acos(std::clamp(object_dot, 0.0, 1.0));
    const bool contacting = state_.segment(178, 16).sum() > 0.0;
    const double contact_score = expected_contact_ > 0.5 ? double(contacting) : double(!contacting);
    rewards_r_.record("wrist_position", std::exp(-wrist_error / std::pow(0.04, 2)));
    rewards_r_.record("wrist_rotation", 0.5 * std::exp(-std::pow(wrist_angle / 0.4, 2)));
    rewards_r_.record("finger_pose", 0.5 * std::exp(-finger_error / std::pow(0.3, 2)));
    rewards_r_.record("joint_position", std::exp(-joint_error / std::pow(0.04, 2)));
    rewards_r_.record("object_position", 2.0 * std::exp(-object_error / std::pow(0.04, 2)));
    rewards_r_.record("object_rotation", 0.5 * std::exp(-std::pow(object_angle / 0.5, 2)));
    rewards_r_.record("contact", 0.25 * contact_score);
    rewards_r_.record("action", -0.02 * action.squaredNorm() / 51.0);
    rewards_r_.record("smoothness", -0.05 * (action - previous_action_).squaredNorm() / 51.0);
    rewards_r_.record("table_penetration", -2.0 * std::min(1.0, std::max(0.0, state_[195] - 0.001) / 0.01));
    rewards_r_.record("object_penetration", -0.5 * std::min(1.0, std::max(0.0, state_[196] - 0.002) / 0.01));
    previous_action_ = action;
    reward_sum_[0] = rewards_r_.sum();
    reward_sum_[1] = 0.0f;
    update_observation();
    return reward_sum_.data();
  }

  void observe(Eigen::Ref<EigenVec> right, Eigen::Ref<EigenVec> left) final {
    right = observation_.cast<float>();
    left.setZero();
  }
  void get_global_state(Eigen::Ref<EigenVec> state) final { state = state_.cast<float>(); }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = 0.0f;
    bool terminal = !state_.allFinite() || (has_initial_state_ && state_[104] < 0.05);
    if (terminal) terminalReward = -10.0f;
    return terminal;
  }

 private:
  static double wrap(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }
  static Eigen::Matrix3d matrix_xyz(const Eigen::Vector3d& angles) {
    return (Eigen::AngleAxisd(angles[0], Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(angles[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(angles[2], Eigen::Vector3d::UnitZ())).toRotationMatrix();
  }
  static double rotation_angle(const Eigen::Vector3d& first, const Eigen::Vector3d& second) {
    return Eigen::AngleAxisd(matrix_xyz(first).transpose() * matrix_xyz(second)).angle();
  }
  void set_friction(double friction) {
    world_->setMaterialPairProp("object", "finger", friction, 0.0, 0.0);
    world_->setMaterialPairProp("object", "table", friction, 0.0, 0.0);
    world_->setMaterialPairProp("finger", "table", friction, 0.0, 0.0);
    world_->setDefaultMaterial(friction, 0.0, 0.0);
  }
  void set_gains(double scale) {
    Eigen::VectorXd kp(51), kd(51);
    kp.head(3).setConstant(400.0 * scale);
    kd.head(3).setConstant(20.0);
    kp.segment(3, 3).setConstant(40.0 * scale);
    kd.segment(3, 3).setConstant(2.0);
    kp.tail(45).setConstant(20.0 * scale);
    kd.tail(45).setConstant(1.0);
    hand_->setPdGains(kp, kd);
  }
  void update_state() {
    Eigen::VectorXd q(51), velocity(51);
    hand_->getState(q, velocity);
    state_.head(51) = q;
    state_.head(3) += base_origin_;
    state_.segment(51, 51) = velocity;
    state_.segment(102, 7) = object_->getGeneralizedCoordinate().e();
    state_.segment(109, 6) = object_->getGeneralizedVelocity().e();
    for (size_t i = 0; i < body_frames_.size(); ++i) {
      Vec<3> point;
      hand_->getFramePosition(body_frames_[i], point);
      state_.segment(115 + 3 * i, 3) = point.e();
    }
    state_.segment(178, 21).setZero();
    for (const auto& contact : hand_->getContacts()) {
      if (contact.skip()) continue;
      if (contact.getPairObjectIndex() == object_->getIndexInWorld()) {
        auto found = contact_mapping_.find(contact.getlocalBodyIndex());
        if (found != contact_mapping_.end()) state_[178 + found->second] = 1.0;
        state_[196] = std::max(state_[196], -contact.getDepth());
        state_[198] += contact.getImpulse().norm();
      } else if (contact.getPairObjectIndex() == table_->getIndexInWorld()) {
        state_[194] += 1.0;
        state_[195] = std::max(state_[195], -contact.getDepth());
      }
    }
    state_[197] = state_[104] - initial_object_[2];
  }
  void update_observation() {
    observation_.head(199) = state_;
    observation_.segment(199, 51) = reference_hand_;
    observation_.segment(250, 7) = reference_object_;
    observation_.segment(257, 63) = reference_joints_;
    observation_[320] = phase_;
    observation_[321] = expected_contact_;
    observation_.segment(322, 51) = reference_velocity_;
    observation_.segment(373, 3) = table_pose_.head(3);
    observation_.segment(376, 3) = table_dimensions_;
    observation_.segment(379, 4) = table_pose_.tail(4);
  }

  ArticulatedSystem* hand_ = nullptr;
  ArticulatedSystem* object_ = nullptr;
  Box* table_ = nullptr;
  double table_height_ = 0.5;
  bool domain_randomization_ = false;
  bool has_initial_state_ = false;
  std::string load_set_;
  std::mt19937 random_;
  Eigen::Vector3d base_origin_;
  Eigen::Vector3d reset_base_origin_;
  Eigen::Vector3d table_dimensions_;
  Eigen::VectorXd table_pose_;
  Eigen::VectorXd state_, observation_, reference_hand_, reference_object_, reference_joints_, reference_velocity_;
  Eigen::VectorXd previous_action_, initial_hand_, initial_hand_velocity_, initial_object_;
  std::vector<raisim::Vec<2>> limits_;
  std::vector<double> nominal_masses_;
  std::vector<raisim::Mat<3, 3>> nominal_inertias_;
  std::unordered_map<size_t, size_t> contact_mapping_;
  std::array<float, 2> reward_sum_{{0.0f, 0.0f}};
  double phase_ = 0.0, expected_contact_ = 0.0;
  const std::array<std::string, 21> body_frames_{{
      "right_wrist_0rz", "right_index1_x", "right_index2_x", "right_index3_x", "right_index_tip",
      "right_middle1_x", "right_middle2_x", "right_middle3_x", "right_middle_tip",
      "right_pinky1_x", "right_pinky2_x", "right_pinky3_x", "right_pinky_tip",
      "right_ring1_x", "right_ring2_x", "right_ring3_x", "right_ring_tip",
      "right_thumb1_x", "right_thumb2_x", "right_thumb3_x", "right_thumb_tip"}};
  const std::array<std::string, 16> contact_bodies_{{
      "right_wrist_rz", "right_index1_z", "right_index2_z", "right_index3_z",
      "right_middle1_z", "right_middle2_z", "right_middle3_z",
      "right_pinky1_z", "right_pinky2_z", "right_pinky3_z",
      "right_ring1_z", "right_ring2_z", "right_ring3_z",
      "right_thumb1_z", "right_thumb2_z", "right_thumb3_z"}};
};

}  // namespace raisim
