# PufferDrive rewards

Overview of configurable rewards and what they do.

## Reward components

### Goal completion rewards

#### **Goal reached (primary)**
- **Type**: discrete
- **Value**: `+1.0`
- **Trigger**: agent reaches within 2.0 meters of goal position The "goal" position is currently taken as the end of it's corresponding log trajectory
- **Description**: used to condition agent to drive to any (x,y)  target
- **Frequency**: once per goal achievement

#### **Goal reached (post-respawn)**
- **Type**: discrete
- **Value**: `reward_goal_post_respawn` (configurable)
- **Trigger**: agent reaches goal after being respawned during episode
- **Description**: potentially reduced reward for goal completion after respawn
- **Range**: typically `[0.0, 1.0]`

### Collision penalties

#### **Vehicle collision (clean)**
- **Type**: discrete
- **Value**: `reward_vehicle_collision` (configurable)
- **Trigger**: agent collides with another vehicle before reaching goal
- **Description**: penalty for hitting other cars, affects "clean collision rate" metric
- **Range**: typically `[-1.0, 0.0]`; set to positive value for encouraging collisions.

#### **Vehicle collision (post-respawn)**
- **Type**: discrete
- **Value**: `reward_vehicle_collision_post_respawn` (configurable, negative)
- **Trigger**: agent collides with vehicle after being respawned
- **Description**: potentially reduced penalty for collisions after respawn
- **Range**: typically `[-1.0, 0.0]`

#### **Off-road collision**
- **Type**: discrete
- **Value**: `reward_offroad_collision` (configurable)
- **Trigger**: agent's bounding box intersects with road edge boundaries
- **Description**: penalty for leaving designated driving areas
- **Range**: typically `[-1.0, 0.0]`

### Human log rewards

#### **Average displacement error (ade)**
- **Type**: continuous
- **Value**: `reward_ade * current_ade` (configurable weight × bounded error)
- **Trigger**: applied every timestep when reference trajectory is available
- **Error range**: `[0.0, 1.0)` (bounded by exponential transform)
- **Transform**: `1.0 - exp(-squared_displacement_error)`
- **Description**: penalizes spatial deviation from reference trajectory
- **Total range**: depends on `reward_ade` weight (typically negative)

#### **Heading error**
- **Type**: continuous
- **Value**: `reward_heading * current_heading_error` (configurable weight × error)
- **Trigger**: applied every timestep when reference trajectory is available
- **Error range**: `[0.0, 1.0)` (bounded by exponential transform)
- **Transform**: `1.0 - exp(-squared_heading_difference)`
- **Description**: penalizes angular deviation from reference heading
- **Total range**: depends on `reward_heading` weight (typically negative)

#### **Speed error**
- **Type**: continuous
- **Value**: `reward_speed * current_speed_error` (configurable weight × error)
- **Trigger**: applied every timestep when reference trajectory is available
- **Error range**: `[0.0, 1.0)` (bounded by exponential transform)
- **Transform**: `1.0 - exp(-squared_speed_difference)`
- **Description**: penalizes deviation from reference speed
- **Total range**: depends on `reward_speed` weight (typically negative)

### Lane alignment
- **Type**: todo
- **Description**: reward for maintaining proper lane positioning and orientation
