from utils import *


# Geometry parameters
r_earth = 6378; # radius of earth in kilometers
r_moon = 1737; # radius of moon in kilometers
distance_earth_moon = 384400; # in kilometers
rot_speed_earth = 0 # 2*pi/(24*3600); # rad/s
rot_speed_moon = 0 # 2 * pi / (27.32 * 24 * 3600); # rad/s
inclination = radians(5.14)
phi_synodic_earth = radians(28.58); # deg 
phi_synodic_moon = radians(6.68); # deg
G = 6.67430e-20; # Gravity constant in km^3*kg^-1*h^-2
g_zero_point = np.array([346019.66, 0, 0])

# # Transformation of earth, moon w.r.t synodic
R_synodic_earth_0 = R.from_euler('y', phi_synodic_earth, degrees=False).as_matrix()
ang_vel_earth = np.dot(R_synodic_earth_0, np.array([0, 0, 1]))*rot_speed_earth
R_synodic_moon_0 = R.from_euler('y', phi_synodic_moon, degrees=False).as_matrix()
ang_vel_moon = np.zeros(3)

# Physical parameters
mass_earth = 5.97219e24  # kg
mass_moon = 7.34767e22  # kg
mass_sc = 49735  # kg (initial mass of spacecraft)
mass_sc_final = 4000  # kg (final mass of spacecraft after fuel consumption)

# Thrust parameters
Ve = 12  # Exhaust velocity in km/s
m_dot = 4000  # Mass flow rate in kg/s

# State variables
phi_earth_space = np.deg2rad(60)
theta_earth_space = np.deg2rad(100)
time_stamp = [0]
theta_synodic_space = np.deg2rad(139.0933)
fire_angle = np.deg2rad(42.2394)

X_sc, V_sc, A_sc, thrust_unit = set_initial_new(
    theta = theta_synodic_space,
    alpha = fire_angle,
    r_earth=r_earth
)

# breakpoint()

# Simulation parameters
time_escape_eath = 0.0 
time_to_zero_gra = 0.0
time_land_moon = 0.0
delta_T = 1; # simulation time step (hour)

# Class Space 
Earth = Planet(
    mass=mass_earth,
    radius=r_earth,
    ang_vel=ang_vel_earth,
    theta=0.0,
    phi=phi_synodic_earth,
    distance_synodic=0,
    incline_synodic=0
)

Moon = Planet(
    mass=mass_moon,
    radius=r_moon,
    ang_vel=ang_vel_moon,
    theta=0.0,
    phi=phi_synodic_moon,
    distance_synodic=distance_earth_moon,
    incline_synodic=inclination
)

## Read file 
file_path = os.path.join(os.getcwd(),"data")
save_file_path = os.path.join(file_path,"scraft.csv")
file = pandas.read_csv(save_file_path)
position = np.array(file['Position'].apply(eval).tolist())
velocity = np.array(file['Velocity'].apply(eval).tolist())
x = position[:,0]
y = position[:,1]
z = position[:,2] 


'''
Time stamp of phase1: 1428
Time stamp of phase2: 465719
Time stamp of phase3: 453293
Time stamp of phase4: 953
'''
max_timestamp = len(x)-1
print(max_timestamp)
final_timestamp = max_timestamp # int(1428 + 465719 - 5)
start_timestamp = 0
show_moon = True

# Create a rotation matrix for a given angle and axis
ax = plt.figure().add_subplot(projection='3d')
ax.cla()

# Spacecraft Frame
ax.plot(x[start_timestamp:final_timestamp],y[start_timestamp:final_timestamp],z[start_timestamp:final_timestamp]) 
scale = Earth.radius*0.5
plot_triad(ax, np.array([x[final_timestamp],y[final_timestamp],z[final_timestamp]]), np.eye(3), scale=scale, tag='', text=False) 
ax.quiver(
    x[final_timestamp],
    y[final_timestamp],
    z[final_timestamp],
    scale * velocity[final_timestamp,0]/np.linalg.norm(velocity[final_timestamp]),
    scale * velocity[final_timestamp,1]/np.linalg.norm(velocity[final_timestamp]),
    scale * velocity[final_timestamp,2]/np.linalg.norm(velocity[final_timestamp]),
    color="c",
)

# Synodic frame
plot_triad(ax, np.zeros(3), np.eye(3), scale=Earth.radius*2, tag='Synodic', text=False) 

# Earth frame 
Earth.visualize(ax, color='b', tag='', text=False)

# Moon Frame
if show_moon:
    plot_triad(ax, g_zero_point, np.eye(3),scale=Earth.radius, tag='zero-g', text=True)
    Moon.visualize(ax, color='r', tag='Moon')

set_equal_scale(ax)
ax.axis('on')
ax.grid(True)
plt.show()