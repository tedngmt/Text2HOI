# GRAB tracking environment interface

`ours_grab_tracking` is a separate GraspXL environment. It uses RaiSim physics,
the original right MANO URDF, a free rigid mug, and a **static** table. It neither
changes the original demo nor makes the mug follow a prescribed trajectory.

All Python arrays passed through the original vectorized wrapper must be
contiguous `float32`, with a leading environment dimension. Rotations use active
world rotations. Euler angles are intrinsic `XYZ`; quaternions are `w,x,y,z`.
The object URDF must have seven generalized coordinates and six velocities.
Collision geometry must preserve the mug cavity and handle opening.

`reset_state(hand51_or_base54, ignored_left51, hand_velocity51, ignored_left_velocity51,
object7_or13)` initializes an episode. `hand51` contains world wrist-joint xyz,
world root Euler angles, and 45 local finger Euler angles. **The wrist position
is MANO joint zero, not the MANO translation parameter.** Object coordinates are
world xyz plus quaternion; optionally append six initial generalized velocities.
The physical hand uses the fixed generic GraspXL shape: personalized GRAB hand
shapes require retargeting and introduce an irreducible shape difference.
The optional 54-column form appends a world xyz origin for the hand's virtual
translation actuators. Production centres this origin at each reference window's
wrist-position bounding-box midpoint. Internal xyz equals world wrist xyz minus
that origin; all exported state and targets remain world coordinates. This
avoids wasting the URDF's `[-0.8,0.8]` m travel range on an off-centre reset.
Split windows with per-axis spans above 1.5 m to retain residual-action margin.

`set_goals_r(object7, joints63, hand51, phase_contact2_or_velocity53)` sets the next tracking
target and refreshes the observation without advancing physics. Object goals
affect the reward and observation only. Joint order is wrist, then four joints
(including the tip) each for index, middle, pinky, ring, thumb. The final two
values are phase and expected hand-object contact, both bounded to `[0,1]`.
The 53-column form appends planned hand generalized velocity51 [m/s, rad/s],
used as the PD velocity target. The original two-column form uses zero desired
velocity for compatibility and the controlled baseline comparison. Velocities
are clamped at 5 m/s translation and 20 rad/s rotation. A target pointing beyond
a joint limit also has its outward velocity removed. Planning velocities should
use wrapped angular differences between adjacent reference frames.

`step(right_actions51, ignored_left_actions51, reward_right, reward_left, done)`
is the original vectorized in-place API. The residual actions are clamped to
`[-1,1]`; scales are 0.03 m translation and 0.20 rad root/finger angles. They are
added to the supplied next reference, then passed through joint limits and PD
actuation. The only feed-forward force supports the hand's own weight. The
object receives no controller force and moves under gravity and contact.

`get_global_state(state199)` exposes:

| Slice | Meaning | Units |
|---|---|---|
| `0:51` | World wrist xyz, root XYZ Euler, finger Euler | m, rad |
| `51:102` | Hand generalized velocities | m/s, rad/s |
| `102:109` | Mug xyz + wxyz quaternion | m, dimensionless |
| `109:115` | Mug generalized velocity | m/s, rad/s |
| `115:178` | 21 world joint positions | m |
| `178:194` | Hand-object contact flags: palm + three links/finger | dimensionless |
| `194` | Hand-table contact count | count |
| `195` | Maximum hand-table contact penetration | m |
| `196` | Maximum hand-object contact penetration | m |
| `197` | Object z displacement from reset | m |
| `198` | Sum of hand-object contact impulse norms | N s |

`observe(right383, left1)` appends reference hand51, reference object7, reference
joints63, phase/contact2, planned velocity51, and tabletop context10 to state199. Planned velocity
occupies columns `322:373`, making the PD tracking input observable to the policy.
Tabletop context is world centre xyz `373:376`, dimensions xyz `376:379`, and
world orientation quaternion wxyz `379:383`.
The Python learner should construct
relative features and normalize them using training observations only.

The original vectorized wrapper automatically resets an environment if its
terminal flag is set. Terminal states here mean numerical failure or mug centre
below 0.05 m; reaching an episode's reference end is handled by the Python
learner. Terminal observations returned by the wrapper are reset observations,
so callers must mask bootstrapping and account for this when recording metrics.

Configuration defaults: `load_set: grab_mug`,
`hand_model_r: rhand_mano_low_mass.urdf`, `table_height: 0.5`,
`enable_table_collision: true`, `domain_randomization: false`, `contact_erp: 30.0`.
Constructor defaults `table_length: 2.0`, `table_width: 1.0`,
`table_thickness: 0.5` preserve the standalone mechanics regression scene.
Production uses the recorded GRAB tabletop dimensions approximately
`0.45001498, 0.54001802, 0.005481` m instead. Call
`add_stage(dimensions3, table_pose7)` **before every episode reset** to place it
using the dataset's scene transform. Dimensions must match the constructor;
RaiSim's box does not expose a resizing operation. The static table is rotated
and translated together with the motion when applying scene augmentation.
This avoids adding the oversized regression table's edges to recorded motions.
Training randomization scales mug mass and inertia jointly by `[0.8,1.2]`,
friction by `[0.6,1.0]`, and proportional gains by `[0.9,1.1]` per episode.
Validation should disable it and use separate seeded perturbation evaluations.

`enable_table_collision: false` exists only to verify the physical collision
regression. Production training and evaluation must retain the default `true`.

The reward combines wrist pose, finger pose, joint positions, mug pose, expected
contact, residual effort/smoothness, and contact-depth penalties. Contact depth
measures the simulator's collision proxies; it is **not** a triangle-level MANO
mesh penetration measurement. Report both where geometry QA is needed.

The bundled RaiSim 1.1.6 returns **negative** contact depth for overlap; this
environment reports `max(0, -getDepth())` as positive penetration. Making the
table static also changes its collision group to 63, so the table's own mask
controls the regression toggle. The local RaiSim header describes ERP as an
apparent-inertia-scaled spring parameter without a well-defined physical unit.
The configured value was checked empirically: a 0.08 m box falls onto z=0.5 m
and settles with centre z=0.54 m.

`../verify_simulator.py` checks that an aggressive downward hand target stops
at the table with collisions enabled and passes through when disabled. The
first verification measured a transient maximum proxy depth of 4.75 mm under
this deliberately infeasible target, so collisions do not imply exact zero
penetration. The same test verifies goal updates leave physical state unchanged
and compares 264 poses across all 44 clips to RaiSim's actual joint frames.
