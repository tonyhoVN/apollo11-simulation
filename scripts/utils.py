from math import sin, cos, tan, atan2, pi, radians, degrees
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from termcolor import colored
import csv
import pandas
import os 

save_file_path = os.path.join(os.getcwd(),"data","scraft.csv")

# Define a helper function for plotting axis triads
def plot_triad(ax, origin, rotation, scale=1, linewidth=2, tag="", text=True):
    """Plots a 3D triad (X, Y, Z axes) at a given origin."""
    # X axis
    x_axis = scale * np.matmul(rotation, np.array([1, 0, 0]))
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        color="r",
        linewidth=linewidth,
    )
    # Y axis
    y_axis = scale * np.matmul(rotation, np.array([0, 1, 0]))
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        y_axis[0],
        y_axis[1],
        y_axis[2],
        color="g",
        linewidth=linewidth,
    )
    # Z axis
    z_axis = scale * np.matmul(rotation, np.array([0, 0, 1]))
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        z_axis[0],
        z_axis[1],
        z_axis[2],
        color="b",
        linewidth=linewidth,
    )

    # Label for the frame
    if text:
        ax.text(origin[0] + x_axis[0], origin[1] + x_axis[1], origin[2] + x_axis[2], f"{tag} X", color="r", fontsize=10)
        ax.text(origin[0] + y_axis[0], origin[1] + y_axis[1], origin[2] + y_axis[2], f"{tag} Y", color="g", fontsize=10)
        ax.text(origin[0] + z_axis[0], origin[1] + z_axis[1], origin[2] + z_axis[2], f"{tag} Z", color="b", fontsize=10)


def change_frame(p_B, v_B, a_B, ang_vel_B, R_A_B, p_A_B, type_vec):
    """
    Transport theorem to convert a vector presented in B frame to presented in A frame

    Parameters
    ----------
    p_B : np.array, optional
        position vector presented in B
    v_B : np.array, optional
        velocity vector presented in B
    ang_vel_B : np.array, optional
        angular velocity of B w.r.t A presented in A
    R_A_B : np.array
        rotation matrix B w.r.t A
    p_A_B : np.array
        translation of B w.r.t A
    type_vec : {'pos','vel','acc','all'}
        type of velocity
    """
    A_p_B = np.matmul(R_A_B, p_B)  # position w.r.t B presented in A
    A_v_B = np.matmul(R_A_B, v_B)
    A_a_B = np.matmul(R_A_B, a_B)
    p_A = A_p_B + p_A_B
    v_A = A_v_B + np.cross(ang_vel_B, A_p_B)
    a_A = (
        A_a_B
        + np.cross(ang_vel_B, np.cross(ang_vel_B, A_p_B))
        + 2 * np.cross(ang_vel_B, A_v_B)
    )

    if type_vec == "pos":
        return p_A
    elif type_vec == "vel":
        return v_A
    elif type_vec == "acc":
        return a_A
    else:
        return (p_A, v_A, a_A)


def set_initial_position(phi, theta, radius, R_synodic_earth, ang_vel_earth):
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    p_earth_space = np.array([x, y, z])
    # Position
    pos, vel, acc = change_frame(
        p_B=p_earth_space,
        v_B=np.zeros(3),
        a_B=np.zeros(3),
        ang_vel_B=ang_vel_earth,
        R_A_B=R_synodic_earth,
        p_A_B=np.zeros(3),
        type_vec="all",
    )
    thrust_unit = np.matmul(R_synodic_earth, np.array([0, 0, 0]))
    return (pos, vel, acc, thrust_unit)


def set_initial_new(theta, alpha, r_earth):
    """
    Set initial position and fire angle.

    Parameters
    ==========
    theta: angle of position
    alpha: fire angle
    r_earth: radius of the earth
    """
    x = r_earth * cos(theta)
    y = r_earth * sin(theta)
    pos = np.array([x, y, 0])
    vel = np.zeros(3)
    acc = np.zeros(3)
    thrust_unit = np.array(
        [cos(theta + pi / 2 - alpha), sin(theta + pi / 2 - alpha), 0]
    )
    return (pos, vel, acc, thrust_unit)


class Planet:
    def __init__(
        self,
        mass=0,
        radius=0,
        ang_vel=np.zeros(3),
        theta=0,
        phi=0,
        distance_synodic: float = 0,
        incline_synodic=0,
    ):
        self.mass = mass
        self.radius = radius
        self.ang_vel = ang_vel
        self.theta = theta
        self.phi = phi
        self.x = np.array([distance_synodic / cos(incline_synodic), 0, 0])
        self.R = np.matmul(
            R.from_euler("y", self.phi).as_matrix(),
            R.from_euler("z", self.theta).as_matrix(),
        )

    def set_state(self, theta):
        self.theta = theta
        self.R = np.matmul(
            R.from_euler("y", self.phi).as_matrix(),
            R.from_euler("z", self.theta).as_matrix(),
        )

    def step_update(self, time_step=0.0):
        theta = self.theta + np.linalg.norm(self.ang_vel) * time_step
        self.set_state(theta)

    def visualize(self, ax: Axes3D, color="b", tag="", text=True):
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 20j]
        earth_x = self.radius * np.cos(u) * np.sin(v)
        earth_y = self.radius * np.sin(u) * np.sin(v)
        earth_z = self.radius * np.cos(v)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                point = np.array([earth_x[i][j], earth_y[i][j], earth_z[i][j]])
                new_point = np.matmul(self.R, point) + self.x
                earth_x[i][j] = new_point[0]
                earth_y[i][j] = new_point[1]
                earth_z[i][j] = new_point[2]

        plot_triad(ax, self.x, self.R, scale=self.radius, tag=tag, text=text)
        ax.plot_surface(
            earth_x,
            earth_y,
            earth_z,
            color=color,
            alpha=0.2,
            edgecolor="k",
        )

class SpaceCraft:
    def __init__(
        self,
        save_file:str,
        mass=49735.0,
        Ve=3.6,
        m_dot=4000.0,
        mass_final=4000.0,
        X_init=np.zeros(3),
        V_init=np.zeros(3),
        A_init=np.zeros(3),
        thrust_unit=np.zeros(3),
    ):
        self.x = X_init
        self.v = V_init
        self.a = A_init
        self.thrust_unit = thrust_unit
        self.m = mass
        self.m_final = mass_final
        self.Ve = Ve
        self.m_dot = -m_dot

        # Define the CSV file path
        self.state_file = save_file
        
        # Open the CSV file in write mode and initialize with headers
        # Check if the CSV file exists, if not create it and initialize with headers
        if not os.path.exists(self.state_file):
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Mass', 'Position', 'Velocity', 'Acceleration'])

        # Open the CSV file in append mode
        self.file = open(self.state_file, mode='a', newline='')
        self.writer = csv.writer(self.file)

    def reset(self, save_file: str, m: float, x: np.array, v: np.array, a: np.array):        
        # Close file in write mode to clear previous data
        self.file.close()

        # Open new file
        self.state_file = save_file
        if not os.path.exists(self.state_file):
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Mass', 'Position', 'Velocity', 'Acceleration'])

        # Open the CSV file in append mode
        self.file = open(self.state_file, mode='a', newline='')
        self.writer = csv.writer(self.file)

        # Reset the spacecraft state to initial values 
        self.set_state(m, x, v, a)

    def set_state(self, m, x, v, a):
        # set sate
        self.m = m
        self.x = x
        self.a = a
        self.v = v
        # save state to file 
        self.writer.writerow([m, x.tolist(), v.tolist(), a.tolist()])
        
    def step_update(
        self,
        G,
        earth: Planet,
        moon: Planet,
        thrust_unit=None,
        m_dot=4000,
        Ve=12,
        time_step=0,
    ):
        # Check distance between spacecraft and planets
        m_current = self.m
        v_current = self.v
        x_current = self.x
        r_earth_sc = x_current - earth.x
        r_moon_sc = x_current - moon.x

        # Calculate thrust

        if thrust_unit is None:
            m_dot_current = 0.0
            F_thrust = np.zeros(3)
        else:
            m_dot_current = -m_dot
            F_thrust = abs(Ve * m_dot_current) * thrust_unit

        # Check total mass of spacecraft
        if float(m_current) <= self.m_final:
            print(colored("Out of energy", "yellow"))
            m_dot_current = 0.0
            F_thrust = np.zeros(3)

        # Calculate gravity force
        F_earth_sp = (
            -G * m_current * earth.mass * r_earth_sc / (np.linalg.norm(r_earth_sc) ** 3)
        )
        F_moon_sp = (
            -G * m_current * moon.mass * r_moon_sc / (np.linalg.norm(r_moon_sc) ** 3)
        )

        # Total force
        force_total = F_earth_sp + F_moon_sp + F_thrust  # - m_dot_current*v_current

        # Update state
        m = m_current + m_dot_current * time_step
        a = force_total / m_current
        v = v_current + a * time_step
        x = x_current + v_current * time_step

        # Check boundary condition:
        # If spacecraft close to surface and velocity toward surface -> normal velocity
        if np.linalg.norm(x - earth.x) <= earth.radius:
            print(colored("Land on Earth", "yellow"))
            v_current = (
                v_current
                - (np.dot(v_current, r_earth_sc))
                * r_earth_sc
                / np.linalg.norm(r_earth_sc) ** 2
            )
            v = v_current + a * time_step
            x = x_current + v_current * time_step
        if (np.linalg.norm(r_moon_sc) <= moon.radius) and (
            np.dot(v_current, r_moon_sc) <= 0
        ):
            v_current = (
                v_current
                - (np.dot(v_current, r_moon_sc))
                * r_moon_sc
                / np.linalg.norm(r_moon_sc) ** 2
            )
            v = v_current + a * time_step
            x = x_current + v_current * time_step

        # Set next state
        self.set_state(m, x, v, a)

    def visualize(self, ax, scale=1, linewidth=2, tag=""):
        # coordinate axis
        x_current = self.x
        v_current_unit = self.v / np.linalg.norm(self.v)
        plot_triad(ax, x_current, np.eye(3), scale, linewidth, tag)
        # velocity vector
        ax.quiver(
            x_current[0],
            x_current[1],
            x_current[2],
            scale * v_current_unit[0],
            scale * v_current_unit[1],
            scale * v_current_unit[2],
            color="c",
            linewidth=linewidth,
        )
        # scatter position
        ax.scatter(x_current[0], x_current[1], x_current[2], color="b")
    
    def __del__(self):
        # Close the file when the object is deleted
        if hasattr(self, 'file'):
            self.file.close()


def set_equal_scale(ax):
    """Sets equal scaling for the X, Y, Z axes."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    max_range = np.max(limits[:, 1] - limits[:, 0]) / 2.0
    midpoints = np.mean(limits, axis=1)

    ax.set_xlim3d([midpoints[0] - max_range, midpoints[0] + max_range])
    ax.set_ylim3d([midpoints[1] - max_range, midpoints[1] + max_range])
    ax.set_zlim3d([midpoints[2] - max_range, midpoints[2] + max_range])


def normalize(vec):
    return vec / np.linalg.norm(vec)

def visualize_system(
    spacecraft_state_file,
    spacecraft,
    earth,
    moon,
    g_zero_point,
    show_earth=True,
    show_moon=False,
    scale_axis=1
):
    ax = plt.figure().add_subplot(projection='3d')
    ax.cla()

    # Spacecraft + trajactory
    file = pandas.read_csv(spacecraft_state_file)
    position = np.array(file['Position'].apply(eval).tolist())
    x = position[:,0]
    y = position[:,1]
    z = position[:,2] 
    # ax = plt.subplot(projection='3d')
    ax.plot(x,y,z)
    spacecraft.visualize(ax, scale=earth.radius*scale_axis, tag='SC')
    
    # Synodic frame
    plot_triad(ax, np.zeros(3), np.eye(3), scale=earth.radius*scale_axis, tag='Synodic') 
    
    # Earth frame 
    if show_earth:
        earth.visualize(ax, color='b', tag='Earth')

    if show_moon:
        plot_triad(ax, g_zero_point, np.eye(3),scale=earth.radius*scale_axis, tag='G_zero')
        moon.visualize(ax,color='r', tag='Moon')
    
    set_equal_scale(ax)
    plt.show()