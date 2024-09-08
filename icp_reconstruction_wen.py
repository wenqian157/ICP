import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import open3d as o3d
import copy

global threshold
global trans_init
global iteration
global my_radius
global my_max_nn
global my_voxel_size
threshold = 1
trans_init = np.eye(4)
iteration = 3000
my_radius = 1 
my_max_nn = 30
my_voxel_size = 0.05


def visualize_point_cloud(pointCloud):
    """Visualizes the point cloud using Open3D

    Args:
        pointCloud (np.array): Nx3 array of N 3D points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointCloud)
    o3d.visualization.draw_geometries([pcd])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    
    o3d.visualization.draw_geometries([source_temp, target_temp])

def save_as_ply(pointCloud, filename):
    """Saves a point in .ply format.
    This can later be opened in CloudCompare.

    Args:
        pointCloud (np.array): a point cloud
        filename (string): name of the file where to store point cloud
    """
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointCloud)
    o3d.io.write_point_cloud(filename, pointCloud)

def read_scans(filename):
    """Reads the scans and respective ground truth transformation from the file

    Args:
        filename (string): Path to the file.

    Returns:
        _(dict, dict, dict): A tuple of values, where:
            pointClouds[i] returns ith Nx3 numpy array of coordinates in Lidar frame of reference,
            rotations[i] returns a 3x3 np.array rotation matrix of scan i in the world reference frame
            translations[i] returns a 3x1 vector of translation of scan i in the world reference frame
    """
    scan_name = "lidar_scans"
    position_name = "positions"
    rotation_name = "rotations"

    hf = h5py.File(filename, "r")
    length = len(hf[scan_name])
    # print("There are ", length, "scans in the dataset.")

    pointClouds = dict()
    rotations = dict()
    translations = dict()

    # print("Reading value..")
    for idx in range(length):
        # print(idx)
        index_str = str(idx)
        lidar = hf[f"{scan_name}/{index_str}"][:]
        position = hf[f"{position_name}/{index_str}"][:]
        rotation = hf[f"{rotation_name}/{index_str}"][:]
        pointClouds[idx] = lidar
        rotations[idx] = rotation
        translations[idx] = position

    return pointClouds, rotations, translations

def align_two_point_clouds(source, target):
    # We suggest using Open3d implementation for ICP.
    # Check this link: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html

    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(reg_p2p.transformation)
    draw_registration_result(source_transformed, target, trans_init)
    evalutate_two_point_clouds(source_transformed, target)
    return

def get_transformation_from_two_point_clouds_p2p(first_pcd, second_pcd):
    reg_p2p = o3d.pipelines.registration.registration_icp(
    first_pcd, second_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))

    return reg_p2p.transformation

def get_transformation_from_two_point_clouds_p2l(source, target, ransac=True, refine=False):
    loss = o3d.pipelines.registration.HuberLoss(k=0.3)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    if ransac:
        trans = get_trans_ransac(source, target, refine)
    else:
        trans = np.eye(4)
    
    reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
                                                      threshold, trans,
                                                      p2l,
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))

    return reg_p2l.transformation

def register_all(point_clouds, ransac=True, refine=False, pose_graph=False):
    first_t_previous = trans_init
    transformed_point_clouds = copy.deepcopy(point_clouds)
    for i in range(len(point_clouds))[1:]:
        previous_t_current = get_transformation_from_two_point_clouds_p2l(point_clouds[i], point_clouds[i-1],
                                                                          ransac=ransac, refine=refine)
        first_t_current = np.dot(first_t_previous, previous_t_current)
        transformed_point_clouds[i].transform(first_t_current)
        first_t_previous = first_t_current
    
    if pose_graph:
        transformed_point_clouds = pose_graph(transformed_point_clouds)

    return transformed_point_clouds

def pose_graph(point_clouds):
    # creat pose graph 
    max_correspondence_distance_coarse = my_voxel_size * 60 
    max_correspondence_distance_fine = max_correspondence_distance_coarse * 0.1
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(point_clouds,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)

    # Optimizing PoseGraph ...
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
    #transform     
    result_pcd = copy.deepcopy(point_clouds)
    for point_id in range(len(result_pcd)):
        result_pcd[point_id].transform(pose_graph.nodes[point_id].pose)

    return result_pcd

def ransac(point_clouds, refine=False):
    first_t_previous = trans_init
    transformed_point_clouds = copy.deepcopy(point_clouds)
    for i in range(len(point_clouds))[1:]:
        previous_t_current = get_trans_ransac(point_clouds[i], point_clouds[i-1], refine)
        first_t_current = np.dot(first_t_previous, previous_t_current)
        transformed_point_clouds[i].transform(first_t_current)
        first_t_previous = first_t_current
    
    return transformed_point_clouds

def evalutate_two_point_clouds(source, target):
    print("Alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    return evaluation

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    loss = o3d.pipelines.registration.HuberLoss(k=0.3)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        p2l)
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        p2l)
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))

            elif abs(target_id - source_id) < 3:
            # else:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
                
    return pose_graph

def display(point_clouds, down=False):
    point_clouds = merge_poind_clouds(point_clouds)
    if down:
        point_clouds = point_clouds.voxel_down_sample(voxel_size = my_voxel_size)
    o3d.visualization.draw_geometries([point_clouds])
    return point_clouds

def numpy_array_to_point_cloud(numpyArray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpyArray)
    return pcd

def filter_point_cloud(pointCloud, min_distance=1.0, max_distance=15.0, z=0.5):
    """Filters the point cloud to remove points closer than `min_distance` and further than `max_distance`.

    Args:
        pointCloud (np.array): Nx3 array of N 3D points
        min_distance (float): Minimum distance threshold
        max_distance (float): Maximum distance threshold

    Returns:
        np.array: Filtered point cloud
    """
    distances = np.linalg.norm(pointCloud, axis=1)
    mask = (distances >= min_distance) & (distances <= max_distance)
    return pointCloud[mask] 

def z_filter(pointCloud, z=0.5):
    z_mask = pointCloud[:, 2] < z
    return pointCloud[z_mask]

def merge_poind_clouds(point_clouds):
    merged_pcd = o3d.geometry.PointCloud()
    for point_cloud in point_clouds:
        merged_pcd += point_cloud

    return merged_pcd

def get_trans_ransac(source, target, refine=False):
    source_fpfh = get_fpfh(source)
    target_fpfh = get_fpfh(target)

    result_ransac = excute_global_registration(source, target,
                                            source_fpfh, target_fpfh,
                                            my_voxel_size)

    if refine:
        result_ransac = refine_registration(source, target, result_ransac,
                                 my_voxel_size)
    
    return result_ransac.transformation


    result_pcd = chain_transform_point_clouds(point_clouds)
    result_pcd = pose_graph(result_pcd)

    result_pcd = merge_poind_clouds(result_pcd)
    result_pcd = voxel_downsampling(result_pcd, my_voxel_size)

    return result_pcd

def preprocess_point_cloud(point_cloud, convert_from_array=True):
    if convert_from_array:
        point_cloud = filter_point_cloud(point_cloud, 2, 15)
        point_cloud = numpy_array_to_point_cloud(point_cloud)
    
    pcd_down = point_cloud.voxel_down_sample(voxel_size = my_voxel_size)

    radius_normal = my_voxel_size * 10
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    return pcd_down
    
def get_fpfh(point_cloud):
    radius_feature = my_voxel_size * 25
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return pcd_fpfh  

def excute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

###################################
def main():
    # Reading the data from the file.
    filename = "design++_halter_construction.hdf5"
    numpyArrays, rotations, translations = read_scans(filename)
    test_numpy_arrays = []
    for i in range(len(numpyArrays)):
        test_numpy_arrays.append(numpyArrays[i])

    # preprocess
    point_arrays = test_numpy_arrays[0: 10]
    point_clouds = [preprocess_point_cloud(point_array) for point_array in point_arrays]

    # register
    point_clouds = register_all(point_clouds, ransac=True, refine=False, pose_graph=False)

    #display
    point_clouds = display(point_clouds)

    #save
    # path = "/Users/yang/GitHub/20240904_AIWorkshop/icp_registration_public/0_5.ply"
    # save_as_ply(point_clouds, path)


if __name__ == "__main__":
    main()
