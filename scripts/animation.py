
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from utils import *

file_path = os.path.join(os.getcwd(),"data")
save_file_path = os.path.join(file_path,"scraft.csv")

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

# trajactory
file_path = os.path.join(os.getcwd(),"data")
save_file_path = os.path.join(file_path,"scraft.csv")
file = pandas.read_csv(save_file_path)
position = np.array(file['Position'].apply(eval).tolist())
velocity = np.array(file['Velocity'].apply(eval).tolist())
x = position[:,0]
y = position[:,1]
z = position[:,2] 

# Timestamp
max_timestamp = len(x)-1
time_stamp_phase1 = 5000
time_stamp_phase2 = max_timestamp - 5000
time_stamp_phase3 = max_timestamp
array1 = np.arange(0, time_stamp_phase1, 100)
array2 = np.arange(time_stamp_phase1+1, time_stamp_phase2, 1000)
array3 = np.arange(time_stamp_phase2+1, time_stamp_phase3, 50)
frames_array = np.concatenate((array1, array2, array3))

# Create a rotation matrix for a given angle and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Animation function to rotate the frame around a given axis
def update(frame_id):
    ax.cla()  # Clear the plot

    # Synodic frame
    if x[frame_id] < 150000:
        scale = Earth.radius*0.5
        plot_triad(ax, np.zeros(3), np.eye(3), scale=Earth.radius*3, tag='Synodic', text=False) 
        Earth.visualize(ax, color='b', tag='', text=False)

    # Moon Frame
    if x[frame_id] >= 150000:
        scale = Earth.radius
        plot_triad(ax, g_zero_point, np.eye(3),scale=Earth.radius, tag='zero-g', text=True)
        Moon.visualize(ax, color='r', tag='Moon')
    
    # Trajectory 
    if frame_id <  30000:
        start_timestamp = 0
    elif frame_id >=30000 and frame_id<time_stamp_phase2: 
        start_timestamp = frame_id - 30000
    else:
        start_timestamp = frame_id - 1000

    ax.plot(x[start_timestamp:frame_id],y[start_timestamp:frame_id],z[start_timestamp:frame_id]) 
    
    # Spacecraft Frame
    plot_triad(ax, np.array([x[frame_id],y[frame_id],z[frame_id]]), np.eye(3), scale=scale, tag='', text=False) 
    ax.quiver(
        x[frame_id],
        y[frame_id],
        z[frame_id],
        scale * velocity[frame_id,0]/np.linalg.norm(velocity[frame_id]),
        scale * velocity[frame_id,1]/np.linalg.norm(velocity[frame_id]),
        scale * velocity[frame_id,2]/np.linalg.norm(velocity[frame_id]),
        color="c",
    )

    set_equal_scale(ax)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames_array, interval=10, repeat=True, repeat_delay=10000)

plt.show()