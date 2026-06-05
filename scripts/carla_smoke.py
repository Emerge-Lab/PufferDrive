import carla
client = carla.Client('localhost', 2000); client.set_timeout(60.0)
print('client', client.get_client_version(), '| server', client.get_server_version())
maps = [m.split('/')[-1] for m in client.get_available_maps()]
print('maps:', sorted(set(maps)))
client.load_world('Town01')
world = client.get_world()
s = world.get_settings(); s.synchronous_mode=True; s.fixed_delta_seconds=0.1
world.apply_settings(s)
bp = world.get_blueprint_library().filter('vehicle.*')[0]
sp = world.get_map().get_spawn_points()[0]
veh = world.spawn_actor(bp, sp); world.tick()
t = veh.get_transform(); v = veh.get_velocity()
print('ego @', round(t.location.x,1), round(t.location.y,1), round(t.location.z,1),
      '| yaw', round(t.rotation.yaw,1), '| vel', round(v.x,2), round(v.y,2))
tls = world.get_actors().filter('traffic.traffic_light')
print('traffic lights:', len(tls), '| tl0 state:', tls[0].get_state() if len(tls) else 'n/a')
veh.destroy()
# restore async so server doesn't hang waiting for ticks
s.synchronous_mode=False; world.apply_settings(s)
print('SMOKE OK')
