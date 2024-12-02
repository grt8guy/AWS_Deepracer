'''
This is a helper module for deepracer_analysis.ipynb.
It contains functions for most of the visualizations and analysis.
'''

# Standard Library Imports
from datetime import datetime
from typing import Any, List, Optional, Tuple
import math
import os
import pkg_resources
import re
import importlib.resources

# Third-Party Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf # type: ignore
# tf.disable_v2_behavior is deprecated, using below settings
tf.compat.v1.disable_v2_behavior
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LineString
from shapely.ops import nearest_points
from tensorflow.python.platform import gfile

LEIDOS_PURPLE = '#850F88'
LEIDOS_DARK = '#24115A'
LEIDOS_GREY = '#BDBEBC'

def is_package_installed(package_name: str) -> bool:
    '''
    Check if package is installed.

    Args:
        package_name: str of package name to check
    
    Returns:
        True if package installed, False otherwise
    '''
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def continuous_to_discrete(
        model_metadata: dict,
        df: pd.DataFrame,
        num_angle_buckets: int = 5, 
        num_speed_buckets: int = 4
    ):
    """
    This function converts a continous action space to a discrete one for easier plotting
    and analysis using binning

    Parameters:
        model_metadata (dict): dictionary of model metada
        df (pd.DataFrame): dataframe of merged simtraces
        num_angle_buckets (int): Number of buckets to break the angle component of the action
            space into
        num_speed_buckets (int): Number of buckets to break the speed component of the action
            space into

    Returns:
        model_metadata (dict): updated dictionary of model metada
        df (pd.DataFrame): updated dataframe of merged simtraces
    """

    if 'action_space_type' in model_metadata and model_metadata['action_space_type'] == 'continuous':
        max_angle = model_metadata['action_space']['steering_angle']['high']
        min_angle = model_metadata['action_space']['steering_angle']['low']

        max_speed = model_metadata['action_space']['speed']['high']
        min_speed = model_metadata['action_space']['speed']['low']

        # Determine which discrete bucket would be the equivalent for the continuous action space
        for index, row in df.iterrows():        
            angle_bucket = math.floor(((row["steer"] - min_angle) / (max_angle-min_angle)) * num_angle_buckets)
            speed_bucket = math.floor(((row["throttle"] - min_speed) / (max_speed-min_speed)) * num_speed_buckets)
            if angle_bucket==num_angle_buckets:
                angle_bucket -= 1
            if speed_bucket==num_speed_buckets:
                speed_bucket -= 1
            df.at[index, "action"] = int(angle_bucket * num_speed_buckets + speed_bucket)
        

        # Convert the model metadata in memory to use the new forced discrete action space
        angle_bucket_size = (max_angle - min_angle) / num_angle_buckets
        angle = min_angle + .5 * angle_bucket_size
        speed_bucket_size = (max_speed - min_speed) / num_speed_buckets
        speed = min_speed + .5 * speed_bucket_size    
        model_metadata['action_space'] = []
        index = 0
        for _ in range(num_angle_buckets):
            for _ in range(num_speed_buckets):
                model_metadata['action_space'].append(
                    {'index': index,
                    'speed': speed,
                    'steering_angle': angle})
                index += 1
                speed += speed_bucket_size
            angle += angle_bucket_size
            speed = min_speed + .5 * speed_bucket_size
    
    return model_metadata, df

def visualize_gradcam_discrete_ppo(sess: tf.compat.v1.Session, rgb_img: np.array, sensor: str, category_index: int = 0,
                                   num_of_actions: int = 5):
    """
    This function generates a Grad-CAM heatmap overlayed on the input image to visualize
    which regions of the image contribute most to the model's decision regarding a
    specific action category.
    
    Visualizes Grad-CAM (Gradient-weighted Class Activation Mapping)
    for Discrete PPO (Proximal Policy Optimization) model.

    Parameters:
        sess (tf.Session): TensorFlow session containing the model.
        rgb_img (np.ndarray): RGB image as a NumPy array.
        sensor (str): Name of the sensor for which input tensor is to be retrieved.
        category_index (int): Index of the action for which to visualize the Grad-CAM heatmap.
        num_of_actions (int): Total number of actions in the model.

    Returns:
        np.ndarray: Overlayed heatmap as a NumPy array.

    """

    img_arr = np.array(rgb_img)
    img_arr = rgb_to_gray(img_arr)
    img_arr = np.expand_dims(img_arr, axis=2)

    x = sess.graph.get_tensor_by_name(f'main_level/agent/main/online/network_0/{sensor}/{sensor}:0')
    y = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    feed_dict = {x:[img_arr]}

    #Get he policy head for clipped ppo in coach
    model_out_layer = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    loss = tf.multiply(model_out_layer, tf.one_hot([category_index], num_of_actions))
    reduced_loss = tf.reduce_sum(loss[0])

    # For front cameras use the below
    conv_output = sess.graph.get_tensor_by_name(
        f'main_level/agent/main/online/network_1/{sensor}/Conv2d_4/Conv2D:0')

    grads = tf.gradients(reduced_loss, conv_output)[0]
    output, grads_val = sess.run([conv_output, grads], feed_dict=feed_dict)
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.sum(weights * output, axis=3)

    ##im_h, im_w = 120, 160##
    im_h, im_w = rgb_img.shape[:2]

    cam = cams[0] #img 0
    
    image = np.uint8(rgb_img[:, :, ::-1] * 255.0) # RGB -> BGR
    cam = cv2.resize(cam, (im_w, im_h)) # zoom heatmap
    cam = np.maximum(cam, 0) # relu clip
    heatmap = cam / np.max(cam) # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # grayscale to color
    cam = np.float32(cam) + np.float32(image) # overlay heatmap
    cam = 255 * cam / (np.max(cam) + 1E-5) ##  Add expsilon for stability
    cam = np.uint8(cam[:, :, ::-1]) # to RGB

    return cam

def load_session(pb_path: str, sensor: str) -> tuple:
    """
    Load a TensorFlow session from a saved model and
    return the session along with input and output tensors.

    Args:
        pb_path (str): Path to the protobuf file containing the saved model.
        sensor (str): Name of the sensor for which input tensor is to be retrieved.

    Returns:
        tuple: A tuple containing TensorFlow session, input tensor, and output tensor.
    """
    # Create a TensorFlow session
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                                    allow_soft_placement=True,
                                    log_device_placement=True))

    # Load the protobuf file
    print("load graph:", pb_path)
    with gfile.FastGFile(pb_path,'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Set the loaded graph as default graph
    sess.graph.as_default()

    # Import the graph definition into the current session
    tf.import_graph_def(graph_def, name='')

    # Get names of all nodes in the graph
    graph_nodes=list(graph_def.node)
    node_names = []
    for node in graph_nodes:
        node_names.append(node.name)

    # Retrieve input and output tensors from the graph
    # For front cameras/stereo camera use the below
    input_tensor = sess.graph.get_tensor_by_name(
        f'main_level/agent/main/online/network_0/{sensor}/{sensor}:0')
    output_tensor = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_1/ppo_head_0/policy:0')

    return sess, input_tensor, output_tensor

def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the luminance method.

    Args:
        rgb_image (numpy.ndarray): Input RGB image represented as a numpy array.

    Returns:
        numpy.ndarray: Grayscale version of the input RGB image.
    """
    # Convert RGB to grayscale using luminance method
    gray_image = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
    return gray_image

## Evaluation RUN
def plot_episode_run(df: pd.DataFrame,
                     center_line: list,
                     inner_border: list,
                     outer_border: list,
                     episode: int) -> None:
    """
    Plot the trajectory of a car during an episode along with track borders.

    Args:
        df (pd.DataFrame): DataFrame containing episode data.
        center_line (list): List of (x, y) coordinates defining the center line of the track.
        inner_border (list): List of (x, y) coordinates defining the inner border of the track.
        outer_border (list): List of (x, y) coordinates defining the outer border of the track.
        episode (int): Episode number to plot.

    Returns:
        None
    """
    # Create a new figure
    fig = plt.figure(1, figsize=(12, 16))
    # Add subplot for plotting
    ax = fig.add_subplot(211)
    ax.axis('off')
    # Plot track borders
    print_border(ax, center_line, inner_border, outer_border)

    # Extract data for the given episode
    episode_data = df[df['episode'] == episode]

    # Plot car trajectory for each step in the episode
    for _, row in episode_data.iterrows():
        x1, y1 = row['x'], row['y']
        car_x2, car_y2 = x1 - 0.02, y1 # End point of car vector
        plt.plot([x1, car_x2], [y1, car_y2], '.', color=LEIDOS_PURPLE) # Plot car position
    
    return fig

def plot_top_laps(
        sorted_indexes: list,
        center_line_points: list,
        inner_border_points: list,
        outer_border_points: list,
        episode_map: dict,
        n_laps: int = 5) -> None:
    """
    Plot the top laps with their trajectories on the track.

    Args:
        sorted_indexes (list): Sorted indexes of top laps.
        center_line_points (list):
            List of (x, y) coordinates defining the center line of the track.
        inner_border_points (list):
            List of (x, y) coordinates defining the inner border of the track.
        outer_border_points (list):
            List of (x, y) coordinates defining the outer border of the track.
        episode_map (dict): Dictionary containing episode data.
        n_laps (int): Number of top laps to plot. Default is 5.

    Returns:
        None
    """
    fig = plt.figure(figsize=(12, 30))

    for i in range(n_laps):
        # Get the index of the lap
        lap_idx = sorted_indexes[i]

        # Retrieve episode data for the lap
        episode_data = episode_map[lap_idx]

        # Add subplot for each lap
        ax = fig.add_subplot(n_laps, 1, i+1)
        ax.axis('off')

        # Plot waypoints and lines for the waypoints
        for waypoints_list in [center_line_points, inner_border_points, outer_border_points]:
            line = LineString(waypoints_list)
            plot_coords(ax, line)
            plot_line(ax, line)
            ax.title.set_text(f'Top {i+1}')

        # Plot the trajectory of the car for each step in the lap
        for j in range(1, len(episode_data)-1):
            x1, y1, action, reward, angle,speed = episode_data[j]
            car_x2, car_y2 = x1 - 0.02, y1
            plt.plot([x1, car_x2], [y1, car_y2], '.', color=LEIDOS_PURPLE)
    return fig

def plot_track(
        df: pd.DataFrame,
        center_line: list,
        inner_border: list,
        outer_border: list,
        track_size: tuple =(500, 800)) -> tuple:
    """
    Plot the track and rewards heatmap.

    Args:
        df (pd.DataFrame): DataFrame containing track data.
        center_line (list): List of (x, y) coordinates defining the center line of the track.
        inner_border (list): List of (x, y) coordinates defining the inner border of the track.
        outer_border (list): List of (x, y) coordinates defining the outer border of the track.
        track_size (tuple): Size of the track plot. Default is (500, 800).

    Returns:
        heatmap (histogram): Heatmap data
        extent (list): the minimum and maximum x and y coordinates in the histogram bins
    """
    all_x = []
    all_y = []
    all_rewards = []

    # Extract x, y, and reward data from DataFrame
    for _, row in df.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        reward = float(row["reward"])
        all_x.append(x)
        all_y.append(y)
        all_rewards.append(reward)

    # Create a 2D histogram for rewards heatmap
    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=50)
    # extent is the minimum and maximum x and y coordinates in the histogram bins
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Create a figure and axis for plotting
    fig = plt.figure(1, figsize=track_size)
    ax = fig.add_subplot(111)
    ax.axis('off')
    # Plot track borders
    print_border(ax, center_line, inner_border, outer_border)

    return heatmap.T, extent

def plot_points(ax: Axes, points: np.ndarray) -> None:
    """
    Plot points on the given axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot points on.
        points (numpy.ndarray): Array of points to plot.

    Returns:
        None
    """
    # Scatter plot the points
    ax.scatter(points[:-1, 0], points[:-1, 1], s=1)

    # Annotate each point with its index
    for i, p in enumerate(points):
        ax.annotate(i, (p[0], p[1]))

def get_track_waypoints(track_name: str) -> np.ndarray:
    """
    Load track waypoints from a file.

    Args:
        track_name (str): Name of the track.

    Returns:
        numpy.ndarray: Array containing track waypoints.
    """
    # Load track waypoints from file
    return np.load(f"track_maps/{track_name}.npy")

def merge_csv_files(output_file_path: str, sim_trace_csvs: list) -> None:
    """
    Merge multiple CSV files into a single CSV file.
    Combines simtrace files to output to location supplied in param

    Args:
        output_file_path (str): Path to the output CSV file.
        sim_trace_csvs (list): List of paths to the input CSV files.

    Returns:
        None
    """
    # Extract IDs from filenames and sort the CSV files
    csvs_with_ids = [(int(os.path.basename(file).split("-")[0]), file) for file in sim_trace_csvs]
    csvs_sorted = sorted(csvs_with_ids, key=lambda csvs_with_ids: csvs_with_ids[0])
    csv_files = [csv_file[1] for csv_file in csvs_sorted]

    header_saved = False
    with open(output_file_path, 'w') as f_out:
        for csv_file in csv_files:
            with open(csv_file) as f_in:
                # Copy header from the first file only
                header = next(f_in)
                if not header_saved:
                    f_out.write(header)
                    header_saved = True
                # Copy lines from the input file to the output file
                for line in f_in:
                    # Add quotes around values enclosed in square brackets
                    line = re.sub(r'(\[[^\]]*\])', r'"\1"', line, flags=re.M)
                    f_out.write(line)

def load_data(file_path: str) -> list:
    """
    Load data from a file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        list: List of extracted data.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # Extract data after "SIM_TRACE_LOG" and split by commas
            if "SIM_TRACE_LOG" in line:
                parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split(",")
                data.append(",".join(parts))
    return data

def convert_to_pandas(data: list, waypts: Optional[Any] = None) -> pd.DataFrame:
    """
    Convert raw data to a pandas DataFrame.

    Args:
        data (List[str]): List of strings containing raw data.
        waypts (Any, optional): Waypoints data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing formatted data.
    """

    episodes_per_iter = 20
    header = ['iteration',
              'episode',
              'steps',
              'x',
              'y',
              'yaw',
              'steer',
              'throttle',
              'action',
              'reward',
              'done',
              'on_track',
              'progress',
              'closest_waypoint',
              'track_len',
              'timestamp',
              'cWp']
    data_dict = {h: [] for h in header}

    #ignore the first two dummy values
    for d in data[2:]:
        parts = d.rstrip().split(",")
        data_dict['parts'].append(parts)

        episode = int(parts[0])
        data_dict['episode'].append(episode)
        data_dict['steps'].append(int(parts[1]))

        x = 100*float(parts[2])
        y =100*float(parts[3])
        data_dict['x'].append(x)
        data_dict['y'].append(y)
        if waypts:
            data_dict['cWp'].append(get_closest_waypoint(x, y, waypts))

        data_dict['yaw'].append(float(parts[4]))
        data_dict['steer'].append(float(parts[5]))
        data_dict['throttle'].append(float(parts[6]))
        data_dict['action'].append(float(parts[7]))
        data_dict['reward'].append(float(parts[8]))

        done = 0 if 'False' in parts[9] else 1
        data_dict['done'].append(done)

        data_dict['all_wheels_on_track'].append(parts[10])
        data_dict['progress'].append(float(parts[11]))
        data_dict['closest_waypoint'].append(int(parts[12]))
        data_dict['track_len'].append(float(parts[13]))
        data_dict['tstamp'].append(parts[14])
        data_dict['iteration'].append(int(episode / episodes_per_iter) +1)

    df = pd.DataFrame(data_dict)
    return df

def episode_parser(df: pd.DataFrame) -> Tuple:
    '''
    Arrange data per episode.

    Args:
        df (pd.DataFrame): The DataFrame containing episode data.

    Returns:
        Tuple[Dict[int, List[List[float]]], Dict[int, np.ndarray], List[int]]: 
            A tuple containing action map, episode map, and sorted episode indices.
    '''
    action_dict = {} # Action => [x,y,reward]
    episode_dict = {} # Episode number => [x,y,action,reward]

    for _, row in df.iterrows():
        e = int(row['episode'])
        x = float(row['x'])
        y = float(row['y'])
        angle = float(row['steer'])
        ttl = float(row['throttle'])
        action = int(row['action'])
        reward = float(row['reward'])

        episode_dict.setdefault(e, np.empty((0, 6)))
        episode_dict[e] = np.vstack((episode_dict[e], np.array([x, y, action, reward, angle, ttl])))

        action_dict.setdefault(action, [])
        action_dict[action].append([x, y, reward])

    # top laps
    # Calculate total rewards for each episode
    total_rewards = {}
    for e, arr in episode_dict.items():
        total_rewards[e] = np.sum(arr[:, 3])

    # Sort episodes by total rewards
    sorted_episodes = sorted(total_rewards.keys(), key=lambda k: total_rewards[k], reverse=True)

    return action_dict, episode_dict, sorted_episodes

def make_error_boxes(ax: plt.Axes,
                     xdata: np.ndarray,
                     ydata: np.ndarray,
                     xerror: np.ndarray,
                     yerror: np.ndarray,
                     facecolor: str = 'r',
                     edgecolor: str = 'r',
                     alpha: float = 0.3) -> int:
    """
    Add error boxes to a plot representing data points with error bars.

    Args:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        xdata (np.ndarray): The x-coordinates of the data points.
        ydata (np.ndarray): The y-coordinates of the data points.
        xerror (np.ndarray): The errors in the x-direction.
        yerror (np.ndarray): The errors in the y-direction.
        facecolor (str, optional): The face color of the error boxes. Defaults to 'r'.
        edgecolor (str, optional): The edge color of the error boxes. Defaults to 'r'.
        alpha (float, optional): The transparency of the error boxes. Defaults to 0.3.

    Returns:
        int: Always returns 0.
    """

    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    return 0

def vertex_color(polygon: Polygon) -> str:
    """
    Determine the color of vertices based on whether the polygon is simple or not.

    Args:
        polygon (Polygon): The shapely Polygon object.

    Returns:
        str: The color code for vertices based on the simplicity of the polygon.
    """
    color_map = {
        True: '#6699cc',  # Blue for simple polygons
        False: '#ffcc33'  # Yellow for non-simple polygons
    }

    return color_map[polygon.is_simple]

def plot_coords(ax: plt.Axes, line: LineString) -> None:
    """
    Plot the coordinates of the given LineString on the provided Axes.

    Args:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        line (LineString): The LineString object representing the border.

    Returns:
        None
    """
    x, y = line.xy
    ax.plot(x, y, 'o', color=LEIDOS_GREY)

def plot_bounds(ax: plt.Axes, polygon: Polygon) -> None:
    """
    Plot the boundary of a Polygon on the given matplotlib Axes.

    Args:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        polygon (Polygon): The shapely Polygon object whose boundary is to be plotted.

    Returns:
        None
    """
    x, y = zip(*list((p.x, p.y) for p in polygon.boundary))
    ax.plot(x, y, '.', color=LEIDOS_DARK, zorder=1)

def plot_line(ax: plt.Axes, line: LineString) -> None:
    """
    Plot the line represented by the given LineString on the provided Axes.

    Args:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        line (LineString): The LineString object representing the border.

    Returns:
        None
    """
    x, y = line.xy
    ax.plot(x, y, color=LEIDOS_DARK, alpha=0.7, linewidth=1.5, solid_capstyle='round', zorder=2)

def print_border(ax: plt.Axes,
                 waypoints: List[Tuple[float, float]],
                 inner_border_waypoints: List[Tuple[float, float]],
                 outer_border_waypoints: List[Tuple[float, float]]) -> None:
    """
    Plot the borders represented by the given waypoints on the provided Axes.

    Args:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        waypoints (List[Tuple[float, float]]):
            List of (x, y) coordinates of the border.
        inner_border_waypoints (List[Tuple[float, float]]):
            List of (x, y) coordinates of the inner border.
        outer_border_waypoints (List[Tuple[float, float]]):
            List of (x, y) coordinates of the outer border.

    Returns:
        None
    """
    # Plot waypoints and lines for the waypoints
    for waypoints_list in [waypoints, inner_border_waypoints, outer_border_waypoints]:
        line = LineString(waypoints_list)
        plot_coords(ax, line)
        plot_line(ax, line)

def get_closest_waypoint(x: float, y: float, waypoints: List[Tuple[float, float]]) -> int:
    """
    Find the index of the closest waypoint to the given coordinates.

    Args:
        x (float): X-coordinate.
        y (float): Y-coordinate.
        waypoints (List[Tuple[float, float]]): List of (x, y) coordinates of waypoints.

    Returns:
        int: Index of the closest waypoint.
    """
    closest_index = 0
    min_distance = float('inf')

    for index, (waypoint_x, waypoint_y) in enumerate(waypoints):
        distance = math.sqrt((waypoint_x - x) ** 2 + (waypoint_y - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_index = index

    return closest_index

def plot_grid_world(
        episode_df: pd.DataFrame,
        inner: List[Tuple[float, float]],
        outer: List[Tuple[float, float]],
        scale: float = 1.0, plot: bool = True
        )-> Tuple:
    """
    Plot a scaled version of the lap, along with throttle taken at each position.

    Args:
        episode_df (pd.DataFrame): DataFrame containing episode data.
        inner (List[Tuple[float, float]]): List of inner boundary coordinates.
        outer (List[Tuple[float, float]]): List of outer boundary coordinates.
        scale (float, optional): Scaling factor. Defaults to 1.0.
        plot (bool, optional): Whether to plot. Defaults to True.

    Returns:
        Tuple[float, float, List[Tuple[float, float, float, float, float, float]]]:
            Lap time, average throttle, and lap statistics.
    """
    stats = []
    # Scale inner and outer boundary coordinates
    outer = [(val[0] / scale, val[1] / scale) for val in outer]
    inner = [(val[0] / scale, val[1] / scale) for val in inner]

    # Calculate maximum x and y coordinates from the outer boundary
    max_x = int(np.max([val[0] for val in outer]))
    max_y = int(np.max([val[1] for val in outer]))
    print(f'{max_x=}, {max_y=}')

    # Initialize grid with zeros
    grid = np.zeros((max_x+1, max_y+1))

    # create shapely ring for outter and inner boundaryies
    outer_polygon = Polygon(outer)
    inner_polygon = Polygon(inner)
    print(f'Outer polygon length = {(outer_polygon.length / scale):.2f} (meters)')
    print(f'Inner polygon length = {(inner_polygon.length / scale):.2f} (meters)')

    # Calculate total distance covered during the lap
    dist = 0.0
    for i in range(1, len(episode_df)):
        dist += math.sqrt(
            (episode_df['x'].iloc[i] - episode_df['x'].iloc[i-1])**2 + \
            (episode_df['y'].iloc[i] - episode_df['y'].iloc[i-1])**2)
    dist /= 100.0

    # Calculate lap time
    t0 = datetime.fromtimestamp(float(episode_df['timestamp'].iloc[0]))
    t1 = datetime.fromtimestamp(float(episode_df['timestamp'].iloc[len(episode_df) - 1]))
    lap_time = (t1-t0).total_seconds()

    # Calculate average throttle, maximum throttle, minimum throttle, and velocity
    average_throttle = np.nanmean(episode_df['throttle'])
    max_throttle = np.nanmax(episode_df['throttle'])
    min_throttle = np.nanmin(episode_df['throttle'])
    velocity = dist/lap_time

    print(f'Distance, lap time = {dist:.2f} (meters), {lap_time:.2f} (sec)')
    print(f'Average throttle, velocity = {average_throttle:.2f} (Gazebo), {velocity:.2f} (meters/sec)')

    # Store lap statistics
    stats.append((dist, lap_time, velocity, average_throttle, min_throttle, max_throttle))

    # Plot the lap if plot parameter is True
    if plot is True:
        for y in range(max_y):
            for x in range(max_x):
                point = Point((x, y))

                # this is the track
                if (not inner_polygon.contains(point)) and (outer_polygon.contains(point)):
                    grid[x][y] = -1.0

                # Find DataFrame slice that fits into this grid cell
                df_slice = episode_df[
                    (episode_df['x'] >= (x - 1) * scale) & (episode_df['x'] < x * scale) & \
                    (episode_df['y'] >= (y - 1) * scale) & (episode_df['y'] < y * scale)]

                if len(df_slice) > 0:
                    #average_throttle = np.nanmean(df_slice['throttle'])
                    grid[x][y] = np.nanmean(df_slice['throttle'])

        # Plot the grid
        fig = plt.figure(figsize=(7,7))
        imgplot = plt.imshow(grid)
        plt.colorbar(orientation='vertical')
        plt.title(f'Lap time (sec) = {lap_time:.2f}')
        #plt.savefig('grid.png')

    return lap_time, average_throttle, stats

def get_distance_to_center_stats(episode: np.array, center_line: np.array) -> tuple:
    """Calculates the Euclidean distance of points in an episode to a center line then computes several stats.
    - mean
    - std
    - median

    This function takes an array of points (episode) and a center line, then calculates
    the Euclidean distance from each point to the nearest point on the center line.
    It returns the mean of these distances.

    Args:
        episode (np.array): A 2D numpy array of shape (n, 2) representing points in an episode,
                            where each row is an (x, y) coordinate.
        center_line (np.array): A 2D numpy array of shape (m, 2) representing the center line,
                                where each row is an (x, y) coordinate.

    Returns:
        tuple: mean, std, and median distance from the points in the episode to the center line.

    Examples:
        >>> import numpy as np
        >>> episode = np.array([(1, 2), (2, 3), (3, 4)])
        >>> center_line = np.array([(0, 0), (5, 5)])
        >>> get_mean_distance_to_center(episode, center_line)
        0.7071067811865476

        >>> import numpy as np
        >>> episode = episode_dict[100]
        >>> center_line = np.array([(0, 0), (5, 5)])
        >>> get_mean_distance_to_center(episode, center_line)
        0.7071067811865476
    """

    center_line = LineString(center_line)

    distances = []
    for row in episode:
        x = row[0]
        y = row[1]

        current_point = Point(x, y)

        # The nearest_points function finds the nearest pair of points between the two geometries provided. The first
        # element in the returned tuple is the nearest point on the arbitrary point, and the second element is the nearest
        # point on the Shapely geometric object.
        nearest_centerpoint = nearest_points(current_point, center_line)[1]

        distance_to_center = current_point.distance(nearest_centerpoint)
        distances.append(distance_to_center)

    # Calculate averages
    np_distances = np.array(distances)
    mean_distances = np_distances.mean()
    std_distances = np_distances.std()
    median_distances = np.median(np_distances)
    return {'mean-distance-to-center': mean_distances, 'std-distance-to-center': std_distances, 'median-distance-to-center': median_distances}


def get_timesteps_outside_track(episode: np.array, outer_border: np.array, inner_border: np.array) -> np.array:
    """Counts the number of timesteps where points in an episode are outside the track defined by outer and inner
    borders.

    This function takes an array of points (episode) representing positions at each timestep, and two arrays of points
    representing the outer and inner borders of a track. It calculates how many of those timesteps have points that
    are outside the defined track.

    Args:
        episode (np.array): A 2D numpy array of shape (n, 2) representing points at each timestep,
                            where each row is an (x, y) coordinate.
        outer_border (np.array): A 2D numpy array of shape (m, 2) representing the outer border of the track,
                                 where each row is an (x, y) coordinate.
        inner_border (np.array): A 2D numpy array of shape (k, 2) representing the inner border of the track,
                                 where each row is an (x, y) coordinate.

    Returns:
        int: The number of timesteps where the points are outside the track defined by the outer and inner borders.

    Example:
        >>> episode = np.array([[1, 1], [3, 3], [5, 5], [0, 0]])
        >>> outer_border = np.array([[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]])
        >>> inner_border = np.array([[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]])
        >>> get_timesteps_outside_track(episode, outer_border, inner_border)
        2
    """

    timesteps_outside_track = 0

    border_poly = Polygon(outer_border, [inner_border])

    for row in episode:
        x = row[0]
        y = row[1]

        xy_point = Point(x, y)

        # Check if vehicle is outside borders
        if not border_poly.contains(xy_point):
            timesteps_outside_track += 1

    return timesteps_outside_track


