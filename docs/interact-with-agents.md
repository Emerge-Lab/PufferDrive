## Drive with trained agents

You can take manual control of an agent in the demo by holding **LEFT SHIFT** and using the keyboard controls.

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
- **TAB** - Switch between agents
- **SPACE** - First-person camera view
- **Mouse Drag** - Pan camera
- **Mouse Wheel** - Zoom

### Classic dynamics model

- **UP/W** - Increase acceleration
- **DOWN/S** - Decrease acceleration (brake)
- **LEFT/A** - Steer left
- **RIGHT/D** - Steer right

With the `classic` dynamics model, the controls use accel_delta and steer_delta which increment/decrement the action indices each frame you press the key. So if you tap W multiple times, you go from neutral acceleration (index 3) → 5 → 6 (max acceleration). The action index changes gradually with key presses.

### Jerk dynamics model

- **UP/W** - Accelerate (+4.0 jerk)
- **DOWN/S** - Brake (-15.0 jerk)
- **LEFT/A** - Turn left (+4.0 lateral jerk)
- **RIGHT/D** - Turn right (-4.0 lateral jerk)

With the `jerk` dynamics model, pressing a key directly sets the jerk value; there's no gradual increment. Pressing W always sets jerk to +4.0, regardless of how long you hold it.
