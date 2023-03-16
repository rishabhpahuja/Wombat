
import sys
sys.path.append('../')
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import cv2

class Annotation:
    def __init__(self, anno_id, name, width, height, labels, polygons):
        """
        Create an annotation object

        Parameters
        ----------
        anno_id : int
            ID number of the current image and it's annotations
        name : str
            File name of the image containing the annotations. Note, it does
            not include the path to the image, just the file name.
        width : int
            Width of the image
        height : int
            Height of the image
        labels : list
            List of strings containing component labels. Each entry corresponds
            to a Shapely polygon
        polygons : list
            List of Shapely polygons outlining the component

        Returns
        -------
        None.

        """
        self.id = anno_id
        self.name = name
        self.width = width
        self.height = height
        self.labels = labels
        self.polygons = polygons
        self.warped_polygons = polygons.copy()


def load_CVAT(filename):
    """
    Parses annotations from a .xml file in CVAT 1.1 format.

    Parameters
    ----------
    filename : str
        Path to annotation file

    Returns
    -------
    objects : list
        List of Annotation objects

    """
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []
    for obj in root.findall('image'):

        anno_id = int(obj.get('id'))
        name = obj.get('name')
        width = int(obj.get('width'))
        height = int(obj.get('height'))

        labels = []
        polygons = []
        # Iterate through all of the polygons in the image
        for poly in list(obj):
            labels.append(poly.get('label'))
            s = poly.get('points').split(';')
            points = np.zeros((len(s), 2))
            for i, pair in enumerate(s):
                points[i] = np.array(pair.split(','), dtype=float)
            polygons.append(Polygon(points))

        a = Annotation(anno_id, name, width, height, labels, polygons)
        objects.append(a)
    return objects


def save_CVAT(input_file, annotations, output_file):

    # if output_file[-4:0] != 'xml':
    #     output_file += '.xml'

    # tree = ET.parse(input_file)
    # root = tree.getroot()
    # TODO: Incomplete
    # Converting the xml data to byte object,
    # for allowing flushing data to file
    # stream
    # new_xml = ET.tostring(tree)
    # with open(output_file, 'wb') as f:
    #     f.write(new_xml)
    raise NotImplementedError()

def load_LabelMe(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    anno_id = 0
    name = root.find('filename').text
    width = int(root.find('imagesize')[1].text)
    height = int(root.find('imagesize')[0].text)

    polygons = []
    labels = []
    for obj in tree.findall('object'):
        labels.append(obj.find('name').text)

        point_elements = obj.find('polygon').findall('pt')
        points = np.zeros((len(point_elements), 2))

        for i, pair in enumerate(point_elements):
            points[i] = float(pair[0].text), float(pair[1].text)

        polygons.append(Polygon(points))

    a = Annotation(anno_id, name, width, height, labels, polygons)
    return a


def compute_annotation_mask(width, height, polygons, num_seg=1000, save=False):
    """


    Parameters
    ----------
    width : int
        DESCRIPTION.
    height : int
        DESCRIPTION.
    polygons : TYPE
        DESCRIPTION.
    num_seg : TYPE, optional
        DESCRIPTION. The default is 1000.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    labels_img : TYPE
        DESCRIPTION.

    """
    # Draw grid
    labels_img = create_grid(width, height, num_seg=num_seg, thickness=1)

    # Add polygons to the image
    for component in polygons:
        pts = (np.array([[int(cx), int(cy)] for cx,cy in list(component.exterior.coords)]))
        cv2.fillPoly(labels_img, pts=[pts], color=(200,0,0))

    if save:
        cv2.imwrite('cv2_labels_deformed.png', labels_img)

    return labels_img

def compute_segmentation_mask(width, height, polygons, labels, component_colors):

    labels_img = np.zeros((height, width, 3))
    # Add polygons to the image
    for i in range(len(polygons)):
        component, name = polygons[i], labels[i]
        pts = (np.array([[int(cx), int(cy)] for cx,cy in list(component.exterior.coords)]))
        cv2.fillPoly(labels_img, pts=[pts], color=component_colors[name])

    return labels_img


def create_grid(width, height, num_seg, line_color=(255, 0, 0), thickness=2):
    """
    Create an image of a grid

    Parameters
    ----------
    height : int
        Height of the image of the grid
    width : int
        Width of the image of the grid
    num_seg : int
        Number of grid squares to have
    line_color : tuple, optional
        Color of the grid lines. The default is (255, 0, 0).
    thickness : int, optional
        Thickness of the grid lines. The default is 2.

    Returns
    -------
    img : ndarray
        Image of a grid
        +--+--+
        |--+--|
        +--+--+
    """
    if line_color == (0,0,0):
        img = np.ones((height,width)) * 255
    else:
        img = np.ones((height,width,3)) * 255

    line_type = cv2.LINE_AA

    # Compute pixel area of the image
    area = img.shape[0] * img.shape[1]
    # Find per-segment area (i.e. area of a single grid square)
    segment_area = area / num_seg
    # Compute side length of each grid square
    segments_side = np.sqrt(segment_area)
    # Store the indices that mark the boundaries of each grid square
    row_seg = np.ceil(img.shape[0] / segments_side).astype(int)
    col_seg = np.ceil(img.shape[1] / segments_side).astype(int)
    # These are the indices of the lines that divide the image into a grid
    row_ind = np.linspace(0, img.shape[0], num=row_seg, dtype=int)
    col_ind = np.linspace(0, img.shape[1], num=col_seg, dtype=int)

    # Draw all the horizontal lines
    for i in row_ind:
        cv2.line(img, (0, i), (img.shape[1], i), color=line_color, lineType=line_type, thickness=thickness)
    # Draw all the vertical lines
    for j in col_ind:
        cv2.line(img, (j, 0), (j, img.shape[0]), color=line_color, lineType=line_type, thickness=thickness)
    return img

# Domain randomization functions
def deform(polygon, max_disp):
    """


    Parameters
    ----------
    polygon : TYPE
        DESCRIPTION.
    max_disp : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    """
    Deviates each point in a polygon by a random deviation
    The deviation for each axis is sampled from the uniform distribution [-max_disp, max_disp]
    """
    x,y = polygon.exterior.coords.xy

    x_def = x + (np.random.rand(len(x)) - 0.5)*max_disp*2
    y_def = y + (np.random.rand(len(y)) - 0.5)*max_disp*2

    vertex_list = [(x_def[i], y_def[i]) for i in range(len(x))]

    return Polygon(vertex_list)


def displace(polygon, del_x, del_y):
    """


    Parameters
    ----------
    polygon : TYPE
        DESCRIPTION.
    del_x : TYPE
        DESCRIPTION.
    del_y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x,y = polygon.exterior.coords.xy

    x_disp = np.array(x) + del_x
    y_disp = np.array(y) + del_y

    vertex_list = [(x_disp[i], y_disp[i]) for i in range(len(x))]

    return Polygon(vertex_list)


def rotate(polygon, rot):
    """
    Rotates the polygon about its centroid by rot (radians)


    Parameters
    ----------
    polygon : TYPE
        DESCRIPTION.
    rot : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    x, y = polygon.exterior.coords.xy
    centroidX, centroidY = np.mean(x), np.mean(y)

    x_t, y_t = x - centroidX, y - centroidY
    s, c = np.sin(rot), np.cos(rot)
    R = np.array([[c,-s],[s,c]])

    points_new = R @ np.array([x_t, y_t])

    vertex_list = [points_new[:,i]+[centroidX,centroidY] for i in range(points_new.shape[1])]
    return Polygon(vertex_list)


def expand(polygon, exp):
    """
    Expands the polygon (1+exp) times

    Parameters
    ----------
    polygon : TYPE
        DESCRIPTION.
    exp : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x,y = polygon.exterior.coords.xy
    centroidX, centroidY = np.mean(x), np.mean(y)

    x_t, y_t = x - centroidX, y - centroidY

    x_exp = x_t*(1+exp)+centroidX
    y_exp = y_t*(1+exp)+centroidY

    vertex_list = [(x_exp[i], y_exp[i]) for i in range(len(x))]

    return Polygon(vertex_list)


def manipulate_polygon_list(polygon_id, original_polygons, labels, deformation_func, params):
    """
    Runs deformation_func(*params) on the one of the original polygons

    Parameters
    ----------
    original_polygons : TYPE
        DESCRIPTION.
    polygon_id : TYPE
        DESCRIPTION.
    deformation_func : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    new_polygons : TYPE
        The updated list of polygons

    """
    component = original_polygons[polygon_id]
    component_new = deformation_func(component, *params)

    original_polygons[polygon_id] = component_new

    # new_polygons = original_polygons.copy()

    # Append the updated polygon to the end of the list
    # new_polygons.append(component_new)
    # new_polygons.pop(polygon_id)

    # Append the corresponding label to the end of the list
    # labels.append(labels[polygon_id])
    # labels.pop(polygon_id)  # Remove the old label

    return original_polygons

