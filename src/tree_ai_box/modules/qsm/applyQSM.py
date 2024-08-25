import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import numpy_groupies as npg
import numpy_indexed as npi
from circle_fit import riemannSWFLa
from scipy import interpolate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.cluster import mean_shift


def saveTreeToXML(tree_list, attribute_list, filename):
    """Save the tree structure to an XML file.

    Args:
    ----
        tree_list (list): List of tree branches.
        attribute_list (list): List of attributes for each branch.
        filename (str): Path to save the XML file.
    """
    root = ET.Element("tree")
    branches = {}
    branch_id_counter = 1
    for branch, attrs in zip(tree_list, attribute_list, strict=True):
        branches[branch[0]] = (branch, attrs)

    def create_branch(parent_node, nodes, attrs, level, prev_node_in_parent):
        nonlocal branch_id_counter
        branch = ET.SubElement(
            root, "branch", id=str(branch_id_counter), parent_node=str(parent_node), level=str(level)
        )
        branch_id_counter += 1
        prev_node = prev_node_in_parent
        # Add the parent node to the branch
        if parent_node != -1:
            parent_branch_nodes, parent_branch_attrs = branches[parent_node]
            parent_index = parent_branch_nodes.index(parent_node)
            parent_attr = parent_branch_attrs[parent_index]
            add_node(branch, parent_node, prev_node, level, parent_attr)
            prev_node = parent_node

        for node, attr in zip(nodes, attrs, strict=True):
            node_elem = add_node(branch, node, prev_node, level, attr)
            if node in branches and branches[node][0] != nodes:
                create_branch(node, branches[node][0][1:], branches[node][1][1:], level + 1, prev_node)
                # Move the created sub-branch inside this node
                sub_branch_elem = root.find(f".//branch[@parent_node='{node}']")
                if sub_branch_elem is not None:
                    root.remove(sub_branch_elem)
                    node_elem.append(sub_branch_elem)
            prev_node = node

    def add_node(branch, node, prev_node, level, attr):
        return ET.SubElement(
            branch,
            "node",
            {
                "id": str(node),
                "prev_node": str(prev_node),
                "level": str(level),
                "x": f"{attr[0]:.6f}",
                "y": f"{attr[1]:.6f}",
                "z": f"{attr[2]:.6f}",
                "radius": f"{attr[3]:.6f}",
            },
        )

    create_branch(-1, tree_list[0], attribute_list[0], 1, -1)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with Path(filename).open("w", encoding="utf-8") as file:
        file.write(xml_str)


def removeNanVerticesAndAdjustFaces(vertices, faces):
    """Remove NaN vertices and adjust faces accordingly.

    Args:
    ----
        vertices (list): List of vertices.
        faces (list): List of faces.

    Returns:
    -------
        tuple: Filtered vertices and adjusted faces.
    """
    vertices = np.array(vertices)
    faces = np.array(faces)
    valid_indices = ~np.isnan(vertices).any(axis=1)
    index_map = np.cumsum(valid_indices) - 1
    filtered_vertices = vertices[valid_indices]
    adjusted_faces = index_map[faces]
    valid_faces = adjusted_faces[~(adjusted_faces < 0).any(axis=1)]
    return filtered_vertices.tolist(), valid_faces.tolist()


def saveTreeToObj(tree_centroid_radius, fname, num_segments=50, num_sides=16):
    """Save the tree structure to an OBJ file.

    Args:
    ----
        tree_centroid_radius (list): List of tree branch centroids and radii.
        fname (str): Output file name.
        num_segments (int): Number of segments for tube generation.
        num_sides (int): Number of sides for tube generation.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    for curve_nodes in tree_centroid_radius:
        if len(curve_nodes) <= 3:
            vertices, faces = createLineTube(curve_nodes, num_segments, num_sides)
        else:
            vertices, faces = createBezierTube(curve_nodes, num_segments, num_sides)
        # Remove NaN vertices and adjust faces
        vertices, faces = removeNanVerticesAndAdjustFaces(vertices, faces)
        # Adjust face indices based on the current vertex offset
        adjusted_faces = [
            (f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset, f[3] + vertex_offset) for f in faces
        ]
        all_vertices.extend(vertices)
        all_faces.extend(adjusted_faces)
        vertex_offset += len(vertices)
    # Export to OBJ
    with Path(fname).open("w") as f:
        for vertex in all_vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in all_faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1} {face[3] + 1}\n")


def createBezierTube(nodes, num_segments=50, num_sides=16):
    """Create a Bezier tube from given nodes.

    Args:
    ----
        nodes (list): List of nodes defining the tube.
        num_segments (int): Number of segments for tube generation.
        num_sides (int): Number of sides for tube generation.

    Returns:
    -------
        tuple: Vertices and faces of the generated tube mesh.
    """
    coords = np.array([node[:3] for node in nodes])
    radii = np.array([node[3] for node in nodes])
    # Generate Bezier curve
    t = np.linspace(0, 1, num_segments)
    tck, u = interpolate.splprep(coords.T, s=0)
    points = np.array(interpolate.splev(t, tck)).T
    # Interpolate radii along the curve
    radii_interpolator = interpolate.interp1d(np.linspace(0, 1, len(radii)), radii)
    interpolated_radii = radii_interpolator(t)
    return generateTubeMesh(points, interpolated_radii, num_sides)


def createLineTube(nodes, num_segments=50, num_sides=16):
    """Create a tube mesh along a line defined by nodes.

    Args:
    ----
        nodes (list): List of nodes defining the line.
        num_segments (int): Number of segments along the tube.
        num_sides (int): Number of sides for the tube's cross-section.

    Returns:
    -------
        tuple: Vertices and faces of the generated tube mesh.
    """
    coords = np.array([node[:3] for node in nodes])
    radii = np.array([node[3] for node in nodes])
    # Generate points along the line(s)
    t = np.linspace(0, 1, num_segments)
    points = np.array([np.interp(t, np.linspace(0, 1, len(coords)), coords[:, i]) for i in range(3)]).T
    # Interpolate radii along the line(s)
    interpolated_radii = np.interp(t, np.linspace(0, 1, len(radii)), radii)
    return generateTubeMesh(points, interpolated_radii, num_sides)


def generateTubeMesh(points, radii, num_sides):  # there might be bugs in this function
    """Generate a tube mesh from a series of points and radii.

    Args:
    ----
        points (np.array): Array of points along the tube's center line.
        radii (np.array): Array of radii corresponding to each point.
        num_sides (int): Number of sides for the tube's cross-section.

    Returns:
    -------
        tuple: Vertices and faces of the generated tube mesh.
    """
    vertices = []
    faces = []
    num_segments = len(points)
    for i in range(num_segments):
        direction = points[i + 1] - points[i] if i < num_segments - 1 else points[i] - points[i - 1]
        # Create orthonormal basis
        z_axis = direction / np.linalg.norm(direction)
        x_axis = np.array([1, 0, 0])
        if np.allclose(z_axis, x_axis):
            x_axis = np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        # Generate circle points
        for j in range(num_sides):
            angle = 2 * math.pi * j / num_sides
            x = radii[i] * math.cos(angle)
            y = radii[i] * math.sin(angle)
            point = points[i] + x * x_axis + y * y_axis
            vertices.append(point)
        # Generate faces
        if i < num_segments - 1:
            for j in range(num_sides):
                j_next = (j + 1) % num_sides
                v1 = i * num_sides + j
                v2 = i * num_sides + j_next
                v3 = (i + 1) * num_sides + j_next
                v4 = (i + 1) * num_sides + j
                faces.append((v1, v2, v3, v4))
    return vertices, faces


def findNearestTarget(graph, target_indices, max_graph_distance=np.inf):
    """Find the nearest target for each node in the graph.

    Args:
    ----
        graph (scipy.sparse.csr_matrix): The graph representation.
        target_indices (array): Indices of target nodes.
        max_graph_distance (float): Maximum allowed graph distance.

    Returns:
    -------
        tuple: Nearest targets, indices, distances, and predecessors.
    """
    # Compute the shortest paths to all targets
    distances, predecessors = dijkstra(
        graph, directed=False, indices=target_indices, return_predecessors=True, limit=max_graph_distance
    )
    distances = distances.T
    min_dist_idx = np.argmin(distances, axis=1, keepdims=True)
    nearest_distances = np.take_along_axis(distances, min_dist_idx, axis=1)[:, 0]
    nearest_targets = target_indices[min_dist_idx[:, 0]]
    filter_ind = np.isinf(nearest_distances)
    nearest_targets = nearest_targets[~filter_ind]
    nearest_idx = np.where(~filter_ind)[0]
    nearest_distances = nearest_distances[~filter_ind]
    return nearest_targets, nearest_idx, nearest_distances, predecessors


def getConnectivity(points, k=6, max_distance=0.03):
    """Get connectivity between points based on k-nearest neighbors.

    Args:
    ----
        points (np.array): Array of points.
        k (int): Number of nearest neighbors to consider.
        max_distance (float): Maximum distance for connectivity.

    Returns:
    -------
        np.array: Unique pairs of connected segment IDs.
    """
    kdtree = cKDTree(points[:, :3])
    distances, indices = kdtree.query(
        points[:, :3],
        k=k + 1,
        distance_upper_bound=max_distance,
    )
    distances = distances[:, 1:].ravel()
    indices = indices[:, 1:]
    row_idx = np.repeat(np.arange(len(points)), k)
    col_idx = indices.ravel()
    valid = np.isfinite(distances)
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    joint_ind = (
        points[row_idx, -1] != points[col_idx, -1]
    )  # this is to find the boundary between two connected segs because the labels between two segs won't match
    joint_row_idx = row_idx[joint_ind]
    joint_col_idx = col_idx[joint_ind]

    return np.unique(np.transpose([points[joint_row_idx, -1], points[joint_col_idx, -1]]), axis=0)


def getPointwiseClusterDistance(
    pts,
    segs_centroids,
    segs_centroids_stem,
    max_connectivity_search_k,
    max_connectivity_search_distance,
    occlusion_distance_cutoff,
    nn_centroids_k=6,
    weight_min_dist=0.03,
):
    """Calculate pointwise cluster distances and create a graph.

    Args:
    ----
        pts (np.array): Input points.
        segs_centroids (np.array): Segment centroids.
        segs_centroids_stem (np.array): Stem labels for centroids.
        max_connectivity_search_k (int): Maximum k for connectivity search.
        max_connectivity_search_distance (float): Maximum distance for connectivity search.
        occlusion_distance_cutoff (float): Cutoff distance for occlusion.
        nn_centroids_k (int): Number of nearest neighbors for centroids.
        weight_min_dist (float): Minimum weight distance.

    Returns:
    -------
        tuple: Graph, rows, columns, distances, and segment ID pairs set.
    """
    kdtree = cKDTree(segs_centroids[:, :3])
    centroids_nn_distances, centroids_nn_indices = kdtree.query(segs_centroids[:, :3], k=nn_centroids_k + 1)
    # choose to calculate the knn neighbors of each point, and their label difference could be used to update the connectivity matrix#future: could reduce the time by ignoring paris between stem pts
    seg_id_pairs = getConnectivity(
        pts, k=max_connectivity_search_k, max_distance=max_connectivity_search_distance
    )  # mutually connected or not
    seg_id_pairs_set = set(map(tuple, seg_id_pairs))
    seg_id_pairs_set.update(set(map(tuple, seg_id_pairs[:, ::-1])))  # Add reverse pairs

    rows0 = np.repeat(np.arange(len(segs_centroids)), nn_centroids_k)
    cols0 = centroids_nn_indices[:, 1:].ravel()
    distances0 = centroids_nn_distances[:, 1:].ravel()

    # remove those mutual connection between stem nodes, keeping only branch nodes and between branch and stem nodes
    non_stem_ind = ~np.all(np.transpose([segs_centroids_stem[rows0] > 0, segs_centroids_stem[cols0] > 0]), axis=1)
    rows = rows0[non_stem_ind]
    cols = cols0[non_stem_ind]
    distances = distances0[non_stem_ind]
    weights = distances

    n_centroids = len(segs_centroids)
    if_connected = np.array([(r, c) in seg_id_pairs_set for r, c in zip(rows, cols, strict=True)])
    weights[if_connected] = (
        weight_min_dist * weight_min_dist * distances[if_connected] * distances[if_connected]
    )  # To ensure the connection between serveral nodes won't skip the nearest node. In an obtuse triangle, the sum of two squared side lengths is larger than the third side

    weights[weights > occlusion_distance_cutoff] = np.inf
    graph = csr_matrix((weights, (rows, cols)), shape=(n_centroids, n_centroids))
    return graph, rows0, cols0, distances0, seg_id_pairs_set


def updatePointwiseClusterDistance(
    rows0,
    cols0,
    distances0,
    seg_id_pairs_set,
    segs_centroids_conn_stemlabels,
    occlusion_distance_cutoff=0.4,
    weight_min_dist=0.03,
):
    """Update pointwise cluster distances based on new stem labels.

    Args:
    ----
        rows0 (np.array): Initial row indices.
        cols0 (np.array): Initial column indices.
        distances0 (np.array): Initial distances.
        seg_id_pairs_set (set): Set of segment ID pairs.
        segs_centroids_conn_stemlabels (np.array): Updated stem labels for centroids.
        occlusion_distance_cutoff (float): Cutoff distance for occlusion.
        weight_min_dist (float): Minimum weight distance.

    Returns:
    -------
        scipy.sparse.csr_matrix: Updated graph.
    """
    n_centroids = len(segs_centroids_conn_stemlabels)
    # remove those mutual connection between stem nodes
    non_stem_ind = ~np.all(
        np.transpose([segs_centroids_conn_stemlabels[rows0] > 0, segs_centroids_conn_stemlabels[cols0] > 0]), axis=1
    )
    rows = rows0[non_stem_ind]
    cols = cols0[non_stem_ind]
    distances = distances0[non_stem_ind]
    weights = distances
    if_connected = np.array([(r, c) in seg_id_pairs_set for r, c in zip(rows, cols, strict=True)])
    weights[if_connected] = (
        weight_min_dist * weight_min_dist * distances[if_connected] * distances[if_connected]
    )  # To ensure the connection between serveral nodes won't skip the nearest node. In an obtuse triangle, the sum of two squared side lengths is larger than the third side
    weights[weights > occlusion_distance_cutoff] = np.inf
    return csr_matrix((weights, (rows, cols)), shape=(n_centroids, n_centroids))


def branchSegmentation(branch_pts, bandwidth=0.1):
    """Perform branch segmentation using mean shift clustering.

    Args:
    ----
        branch_pts (np.array): Branch points to segment.
        bandwidth (float): Bandwidth parameter for mean shift clustering.

    Returns:
    -------
        tuple: Segment labels and number of clusters.
    """
    resolution = np.array([bandwidth / 2, bandwidth / 2, bandwidth / 2])
    xyz_min = np.min(branch_pts[:, :3], axis=0)
    xyz_max = np.max(branch_pts[:, :3], axis=0)

    block_shape = np.floor((xyz_max[:3] - xyz_min[:3]) / resolution).astype(np.int32) + 1
    block_shape = block_shape[[1, 0, 2]]

    block_x = xyz_max[1] - branch_pts[:, 1]
    block_y = branch_pts[:, 0] - xyz_min[0]
    block_z = branch_pts[:, 2] - xyz_min[2]

    block_ijk = np.floor(
        np.concatenate([block_x[:, np.newaxis], block_y[:, np.newaxis], block_z[:, np.newaxis]], axis=1) / resolution
    ).astype(np.int32)
    block_idx = np.ravel_multi_index((np.transpose(block_ijk)).astype(np.int32), block_shape)
    block_idx_u, block_idx_uidx, block_inverse_idx = np.unique(block_idx, return_index=True, return_inverse=True)

    pts_dec = branch_pts[block_idx_uidx]
    pts_dec = pts_dec[:, :3]

    cluster_centers, labels = mean_shift(pts_dec, bandwidth=bandwidth, bin_seeding=True)
    return labels[block_inverse_idx], len(cluster_centers)


def movingAverage(x, w):
    """Calculate the moving average of a 1D array.

    Args:
    ----
        x (np.array): Input array.
        w (int): Window size for moving average.

    Returns:
    -------
        np.array: Moving average of the input array.
    """
    return np.convolve(x, np.ones(w), "valid") / w


def findStems(
    pts, segs_centroids, segs_centroids_stem_labels, nn_centroids_k=5, weight_min_dist=0.05
):  # pts[x,y,z,stemcls[stem:1],init_segs]
    """Find stem segments in the point cloud.

    Args:
    ----
        pts (np.array): Input points.
        segs_centroids (np.array): Segment centroids.
        segs_centroids_stem_labels (np.array): Stem labels for centroids.
        nn_centroids_k (int): Number of nearest neighbors for centroids.
        weight_min_dist (float): Minimum weight distance.

    Returns:
    -------
        tuple: Stem indices and updated stem labels.
    """
    stempts = pts[pts[:, -2] > 0]
    stem_labels_idx = np.where(segs_centroids_stem_labels > 0)[0]
    segs_centroids_stem = segs_centroids[stem_labels_idx]

    kdtree = cKDTree(segs_centroids_stem[:, :3])
    centroids_nn_distances, centroids_nn_indices = kdtree.query(segs_centroids_stem[:, :3], k=nn_centroids_k + 1)
    # choose to calculate the knn neighbors of each point, and their label difference could be used to update the connectivity matrix#future: could reduce the time by ignoring paris between stem pts
    seg_id_pairs = getConnectivity(stempts, k=nn_centroids_k)  # mutually connected or not
    seg_id_pairs_set = set(map(tuple, seg_id_pairs))
    seg_id_pairs_set.update(set(map(tuple, seg_id_pairs[:, ::-1])))  # Add reverse pairs

    rows = np.repeat(np.arange(len(segs_centroids_stem)), nn_centroids_k)
    cols = centroids_nn_indices[:, 1:].ravel()
    distances = centroids_nn_distances[:, 1:].ravel()

    weights = np.full(len(distances), np.inf)

    n_centroids = len(segs_centroids_stem)
    if_connected = np.array([(r, c) in seg_id_pairs_set for r, c in zip(rows, cols, strict=True)])
    weights[if_connected] = (
        weight_min_dist * weight_min_dist * distances[if_connected] * distances[if_connected]
    )  # To ensure the connection between serveral nodes won't skip the nearest node. In an obtuse triangle, the sum of two squared side lengths is larger than the third side

    graph = csr_matrix((weights, (rows, cols)), shape=(n_centroids, n_centroids))

    start_index = np.argmin(segs_centroids_stem[:, 2])
    distances, predecessors = dijkstra(graph, directed=False, indices=start_index, return_predecessors=True)
    connected_centroids_idx = np.where(~np.isinf(distances))[0]
    end_index = connected_centroids_idx[np.argmax(distances[connected_centroids_idx])]
    stem_path = [end_index]
    stem_path.extend(predecessors[current] for current in stem_path if current >= 0)
    stem_path = stem_path[:-1]
    stem_path = stem_path[::-1]  # from bottom to top

    segs_centroids_stem_labels = np.zeros(len(segs_centroids))
    segs_centroids_stem_labels[stem_labels_idx[stem_path]] = 1

    return stem_labels_idx[stem_path], segs_centroids_stem_labels


def initSegmentation(pts, stem_spacing=0.2, branch_bandwidth=0.1, progress_bar=None):
    """Perform initial segmentation of the point cloud.

    Args:
    ----
        pts (np.array): Input points.
        stem_spacing (float): Spacing parameter for stem segmentation.
        branch_bandwidth (float): Bandwidth parameter for branch segmentation.
        progress_bar (QProgressBar, optional): Progress bar for UI updates.

    Returns:
    -------
        np.array: Initial segment labels.
    """
    if progress_bar:
        progress_bar.setValue(0)
    init_segs = np.zeros(len(pts), dtype=np.int32)
    stemcls = pts[:, -1]
    stem_pts_ind = stemcls - np.min(stemcls) > 0
    stem_pts = pts[stem_pts_ind]

    if progress_bar:
        progress_bar.setValue(10)
    init_segs[stem_pts_ind], n_stem_segs = branchSegmentation(stem_pts, stem_spacing)

    branch_segs_labels, n_branch_segs = branchSegmentation(pts[~stem_pts_ind], bandwidth=branch_bandwidth)
    init_segs[~stem_pts_ind] = branch_segs_labels + n_stem_segs
    if progress_bar:
        progress_bar.setValue(100)
    return init_segs


def refineBranchHead(tree, segs_centroids, max_distance=0.3):
    """Refine the head of branches in the tree structure.

    Args:
    ----
        tree (list): Tree structure.
        segs_centroids (np.array): Segment centroids.
        max_distance (float): Maximum distance for refinement.

    Returns:
    -------
        list: Refined tree structure.
    """
    branch_head_segment = np.array([path[:3] for path in tree if len(path) > 2])
    branch_path_idx = np.array([i for i, path in enumerate(tree) if len(path) > 2])
    branch_head_pts = segs_centroids[branch_head_segment[:, 0]][:, :3]
    branch_head_previous_pts = segs_centroids[branch_head_segment[:, 1]][:, :3]
    branch_head_previous_previous_pts = segs_centroids[branch_head_segment[:, 2]][:, :3]
    branch_head_previous_dir = branch_head_previous_previous_pts - branch_head_previous_pts
    branch_head_previous_dir = (
        branch_head_previous_dir / np.linalg.norm(branch_head_previous_dir, axis=-1)[:, np.newaxis]
    )

    kdtree = cKDTree(branch_head_pts)
    branch_head_nn_distances, branch_head_nn_indices = kdtree.query(
        branch_head_previous_pts, k=4, distance_upper_bound=max_distance
    )
    for i in range(len(branch_head_previous_pts)):
        valid_i = np.where(
            np.all([branch_head_nn_distances[i, :] > 0, (~np.isinf(branch_head_nn_distances[i, :]))], axis=0)
        )[0]  # find head from the second: not too far and not itself
        if len(valid_i) > 0:
            branch_head_vec = branch_head_previous_pts[i] - branch_head_pts[branch_head_nn_indices[i, valid_i]]
            branch_head_dir = branch_head_vec / np.linalg.norm(branch_head_vec, axis=-1)[:, np.newaxis]
            if len(branch_head_dir) > 1:
                branch_head_nn_idx = np.argmin(np.arccos(np.dot(branch_head_dir, branch_head_previous_dir[i])))
            else:
                branch_head_nn_idx = 0
            tree[branch_path_idx[i]][0] = branch_head_segment[branch_head_nn_indices[i, valid_i[branch_head_nn_idx]], 0]
    return tree


def recreateTreePath(
    stems_idx, graph, n_segs_centroids, max_graph_distance, tree=None, init_seg_id=2, segs_labels=None
):  # this needs to be improved
    """Recreate the tree path structure.

    Args:
    ----
        stems_idx (np.array): Indices of stem segments.
        graph (scipy.sparse.csr_matrix): Graph representation.
        n_segs_centroids (int): Number of segment centroids.
        max_graph_distance (float): Maximum graph distance.
        tree (list, optional): Existing tree structure.
        init_seg_id (int): Initial segment ID.
        segs_labels (np.array, optional): Segment labels.

    Returns:
    -------
        tuple: Updated tree, segment labels, and next segment ID.
    """
    # Find nearest targets
    nearest_targets, nearest_idxs, nearest_distances, predecessors = findNearestTarget(
        graph, stems_idx, max_graph_distance
    )
    target_position = {t: i for i, t in enumerate(stems_idx)}
    used_nodes = set(stems_idx)  # Initialize with target nodes
    if tree is None:
        tree = [stems_idx.tolist()]  # tree: stem comes first; bottom node as the first node
    if segs_labels is None:
        segs_labels = np.zeros(n_segs_centroids)
        segs_labels[stems_idx] = 1

    # Sort paths by distance (longest first)
    sorted_indices = np.argsort(nearest_distances)[::-1]
    seg_id = init_seg_id
    for idx in sorted_indices:
        path = []
        current = nearest_idxs[idx]
        target = nearest_targets[idx]
        target_pos = target_position[target]

        while current not in used_nodes:
            path.append(current)
            used_nodes.add(current)
            current = predecessors[target_pos, current]

        if path:
            path.append(current)  # Add the connecting node (either a used node or the target)
            # Future: if the path curvature is too twisted, just cut the path
            tree.append(path[::-1])  # Reverse the path for correct order
            segs_labels[path[::-1][1:]] = seg_id  # from parent branch to branch end
            seg_id += 1
            # # smooth nodes
            # for k in range(3):
            #     segs_centroids[path[::-1],k] = movingAverage(segs_centroids[path[::-1],k],smooth_ma_k)
    return tree, segs_labels, seg_id - 1


def cleanTree(tree, segs_centroids_counts, segs_labels, min_pts=5):
    """Clean the tree structure by removing small segments.

    Args:
    ----
        tree (list): Tree structure.
        segs_centroids_counts (np.array): Count of points in each segment.
        segs_labels (np.array): Segment labels.
        min_pts (int): Minimum number of points for a valid segment.

    Returns:
    -------
        tuple: Cleaned tree and updated segment labels.
    """
    keep_idx = []
    for i, path in enumerate(tree):
        if len(path) < 3 and segs_centroids_counts[path[-1]] <= min_pts:
            segs_labels[path[1:]] = 0.0
            continue
        keep_idx.append(i)
    tree = [tree[idx] for idx in keep_idx]
    return tree, segs_labels


def calculateRadius(pts, tree, segs_centroids, min_r=0.04):  # this needs to be improved
    """Calculate the radius for each node in the tree.

    Args:
    ----
        pts (np.array): Input points.
        tree (list): Tree structure.
        segs_centroids (np.array): Segment centroids.
        min_r (float): Minimum radius.

    Returns:
    -------
        list: Tree structure with calculated radii.
    """
    _, segs_centroids_group = npi.group_by(pts[:, -1].astype(np.int32), np.arange(len(pts[:, -1])))
    tree_centroid_radius = []
    for path in tree:
        path_centroid_radius = []
        segs_centroid_vec = segs_centroids[path[1:]] - segs_centroids[path[:-1]]
        segs_centroid_dir = segs_centroid_vec / np.linalg.norm(segs_centroid_vec, axis=-1)[:, np.newaxis]
        for i, node in enumerate(path[1:]):
            if len(segs_centroids_group[node]) > 4:
                # project to the horizontal plane and circle fitting
                segs_vec = pts[segs_centroids_group[node], :3] - segs_centroids[node, :3]
                segs_prj = np.dot(segs_vec, segs_centroid_dir[i, :3])
                segs_prj_2d = pts[segs_centroids_group[node], :2] - np.outer(segs_prj, segs_centroid_dir[i, :3])[:, :2]
                r0 = np.median(np.sqrt(np.sum(np.power(segs_prj_2d - np.mean(segs_prj_2d, 0), 2), axis=1)))
                xc, yc, r, sigma = riemannSWFLa(segs_prj_2d)
                if sigma / r > 0.3 or r > 0.1:  # only fit circles when the branches or stems are large enough
                    path_centroid_radius.append(
                        np.array([segs_centroids[node, 0], segs_centroids[node, 1], segs_centroids[node, 2], r0])
                    )
                else:
                    path_centroid_radius.append(np.array([xc, yc, segs_centroids[node, 2], np.minimum(r, r0 * 1.2)]))
            else:
                path_centroid_radius.append(
                    np.array([segs_centroids[node, 0], segs_centroids[node, 1], segs_centroids[node, 2], min_r])
                )

        path_centroid_radius.insert(
            0,
            np.array(
                [
                    segs_centroids[path[0], 0],
                    segs_centroids[path[0], 1],
                    segs_centroids[path[0], 2],
                    path_centroid_radius[0][-1],
                ]
            ),
        )
        tree_centroid_radius.append(path_centroid_radius)
    return tree_centroid_radius


def applyQSM(
    pts,
    k_neighbors=6,
    max_graph_distance=40,
    max_connectivity_search_distance=0.03,
    occlusion_distance_cutoff=0.4,
    smooth_ma_k=3,
    progress_bar=None,
    wd=None,
):  # pts[x,y,z,stemcls[stem:1],init_segs]
    """Apply Quantitative Structure Model (QSM) to point cloud data.

    Args:
    ----
        pts (np.array): Input points [x, y, z, stemcls, init_segs].
        k_neighbors (int): Number of neighbors for connectivity.
        max_graph_distance (float): Maximum graph distance.
        max_connectivity_search_distance (float): Maximum distance for connectivity search.
        occlusion_distance_cutoff (float): Cutoff distance for occlusion.
        smooth_ma_k (int): Window size for moving average smoothing.
        progress_bar (QProgressBar, optional): Progress bar for UI updates.
        wd (str, optional): Working directory.

    Returns:
    -------
        tuple: Tree structure, segment centroids, segment labels, and tree centroids with radii.
    """
    if progress_bar:
        progress_bar.setValue(0)

    segs_centroids = npg.aggregate(
        pts[:, -1].astype(np.int32), pts[:, :3], axis=0, func=np.mean
    )  # pts[-1] is init_segs
    segs_centroids_counts = npg.aggregate(pts[:, -1].astype(np.int32), 1)  # pts[-1] is init_segs
    segs_centroids_stem_labels = npg.aggregate(
        pts[:, -1].astype(np.int32), pts[:, -2], axis=0, func=np.max
    )  # pts[-1] is init_segs
    _, segs_centroids_inverse_idx = np.unique(pts[:, -1].astype(np.int32), return_inverse=True)

    # find stems by the longest among the shortest pathes to the bottom point
    stems_idx, segs_centroids_stem_labels = findStems(pts, segs_centroids, segs_centroids_stem_labels)
    graph, rows0, cols0, distances0, seg_id_pairs_set = getPointwiseClusterDistance(
        pts,
        segs_centroids,
        segs_centroids_stem_labels,
        max_connectivity_search_k=k_neighbors,
        max_connectivity_search_distance=max_connectivity_search_distance,
        occlusion_distance_cutoff=occlusion_distance_cutoff * 0.5,
        nn_centroids_k=k_neighbors,
    )

    if progress_bar:
        progress_bar.setValue(50)

    n_segs_centroids = len(segs_centroids)
    tree, segs_labels, n_segs = recreateTreePath(stems_idx, graph, n_segs_centroids, max_graph_distance)

    if progress_bar:
        progress_bar.setValue(70)

    stem_new_ind = segs_labels > 0
    stem_new_idx = np.where(stem_new_ind)[0]

    # repeat the shortest path algorithm to the "less connected" branches
    graph = updatePointwiseClusterDistance(
        rows0,
        cols0,
        distances0,
        seg_id_pairs_set,
        stem_new_ind,
        occlusion_distance_cutoff=occlusion_distance_cutoff,
        weight_min_dist=0.03,
    )
    tree, segs_labels, n_segs = recreateTreePath(
        stem_new_idx, graph, n_segs_centroids, max_graph_distance, tree, init_seg_id=n_segs + 1, segs_labels=segs_labels
    )

    if progress_bar:
        progress_bar.setValue(70)

    tree = refineBranchHead(tree, segs_centroids, max_distance=occlusion_distance_cutoff)
    tree, segs_labels = cleanTree(tree, segs_centroids_counts, segs_labels)
    tree_centroid_radius = calculateRadius(pts, tree, segs_centroids)
    segs_labels = segs_labels[segs_centroids_inverse_idx].astype(np.int32)

    if progress_bar:
        progress_bar.setValue(80)
    return tree, segs_centroids, segs_labels, tree_centroid_radius
