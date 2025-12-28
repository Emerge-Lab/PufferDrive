## Drive with trained agents

You can take manual control of an agent in the simulator by holding **LEFT SHIFT** and using the keyboard controls. When you're in control, the action values displayed on screen will turn **yellow**.

### Local rendering

To launch an interactive renderer, first build:

```bash
bash scripts/build_ocean.sh drive local
```

then launch:

```bash
./drive
```

This will run `demo()` with an existing model checkpoint.

### Controls

**General:**

- **LEFT SHIFT + Arrow Keys/WASD** - Take manual control
- **SPACE** - First-person camera view
- **Mouse Drag** - Pan camera
- **Mouse Wheel** - Zoom

**Classic dynamics model**

- **SHIFT + UP/W** - Increase acceleration
- **SHIFT + DOWN/S** - Decrease acceleration (brake)
- **SHIFT + LEFT/A** - Steer left
- **SHIFT + RIGHT/D** - Steer right

Each key press increments or decrements the action level. For example, tapping W multiple times increases acceleration from neutral (index 3) → 5 → 6 (maximum acceleration). We assume **no friction**, so releasing all keys maintains constant speed and heading.

**Jerk dynamics model**

- **SHIFT + UP/W** - Accelerate (+4.0 m/s³ jerk)
- **SHIFT + DOWN/S** - Brake (-15.0 m/s³ jerk)
- **SHIFT + LEFT/A** - Turn left (+4.0 m/s³ lateral jerk)
- **SHIFT + RIGHT/D** - Turn right (-4.0 m/s³ lateral jerk)

Actions are applied directly when keys are pressed. Pressing W always applies +4.0 m/s³ longitudinal jerk, regardless of how long the key is held.
