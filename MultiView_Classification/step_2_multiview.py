import math
import numpy
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Display.SimpleGui import init_display
from OCC.Display.WebGl import threejs_renderer
import sys
import os
from OCC.Extend.ShapeFactory import translate_shp, rotate_shp_3_axis
import math


def fibonacci_sphere(samples=12, distance=5):
    """
    :param samples: number of views
    :param distance: distance from the center of the object
    :return: # samples points of view around the model
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = (1 - (i / float(samples - 1)) * 2) * distance   # y goes from 1 to -1
        radius = math.sqrt(distance * distance - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
        
        ass = math.sqrt(math.pow(x,2)+math.pow(y,2)+math.pow(z,2) )
        
    return points


def animate_viewpoint2(display, img_name):
    """
    :param img_name: save name of the view
    """
    display.FitAll()
    display.Context.UpdateCurrentViewer()

    cam = display.View.Camera()  # type: Graphic3d_Camera

    center = cam.Center()
    eye = cam.Eye()

    display.View.FitAll()
    eye_ = numpy.array([eye.X(), eye.Y(), eye.Z()])
    center_ = numpy.array([center.X(), center.Y(), center.Z()])
    distance = numpy.linalg.norm(eye_ - center_)

    points = fibonacci_sphere(samples=12, distance=distance)

    for i, point in enumerate(points):

        eye.SetX(point[0]+center_[0])
        eye.SetY(point[1]+center_[1])
        eye.SetZ(point[2]+center_[2])
        cam.SetEye(eye)

        display.View.FitAll()
        display.Context.UpdateCurrentViewer()
        name = img_name.replace(".jpeg", "_"+str(i)+".jpeg")
        display.View.Dump(name)


def make_multiview_dataset(models_dir_path, mvcnn_images_dir_path):
    """
    Generate 12 2D views around of each 3D model of the STEP dataset and save them in the path specified by mvcnn_images_dir_path input
    :param models_dir_path: path of the input step dataset
    :param mvcnn_images_dir_path:  path of the output multi views dataset
    """
    for class_ in os.listdir(models_dir_path):
        if os.path.isdir(models_dir_path + class_):
            for file in os.listdir(models_dir_path + class_ + "/"):
                if file.endswith(".stp"):

                    print("--- Examinating: class:", class_, " - Model:", str(file))

                    if not os.path.exists(mvcnn_images_dir_path + class_ + "/"):
                        os.mkdir(mvcnn_images_dir_path + class_ + "/")

                    img_name = mvcnn_images_dir_path + class_ + "/" + file.replace(".stp", ".jpeg")

                    if not os.path.exists(img_name.replace(".jpeg", "_0.jpeg")):

                        step_reader = STEPControl_Reader()
                        status = step_reader.ReadFile(models_dir_path + class_ + "/" + file)

                        if status == IFSelect_RetDone: # check status
                            failsonly = False
                            step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
                            step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)

                            ok = step_reader.TransferRoot(1)
                            _nbs = step_reader.NbShapes()
                            aResShape = step_reader.Shape(1)
                        else:
                            print("Error: can't read file.")
                            sys.exit(0)

                        display, start_display, add_menu, add_function_to_menu = init_display()
                        display.DisplayShape(aResShape, update=True)

                        animate_viewpoint2(display=display, img_name=img_name)


if __name__ == "__main__":
    path_step = "path1"
    path_multiview = "path2"

    make_multiview_dataset(path_step, path_multiview)



