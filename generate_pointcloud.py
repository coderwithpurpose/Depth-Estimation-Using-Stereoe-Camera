from __future__ import print_function
import numpy as np
import cv2
import pptk



ply_header = '''ply

format ascii 1.0

element vertex %(vert_num)d

property float x

property float y

property float z

property uchar red

property uchar green

property uchar blue 


end_header

'''


def write_ply(fn, verts, colors):

    verts = verts.reshape(-1, 3)

    colors = colors.reshape(-1, 3)

    verts = np.hstack([verts, colors])

    with open(fn, 'wb') as f:

        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))

        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])
    ply_header = '''ply

		format ascii 1.0

		element vertex %(vert_num)d

		property float x

		property float y

		property float z

		property uchar red

		property uchar green

		property uchar blue

		end_header

		'''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')

def main(imgL_BGR, disp, Q):

    # disparity range is tuned for 'aloe' image pair
    min_disp = 0
    print('generating 3d point cloud...',)
    rgb = pptk.rand(100, 3)
    points = cv2.reprojectImageTo3D(disp, Q)
    w,l,c = (np.shape(points))

    # print('old mpoint cloud',pts)
    # # print('new point cloud ', pts[~np.isinf((pts)).any(axis=1)])
    colors = cv2.cvtColor(imgL_BGR, cv2.COLOR_BGR2RGB)
    points = np.reshape(points,(w*l,c))
    rgb =  np.reshape(colors,(w*l,c))
    rgb= rgb[~np.isinf((points)).any(axis=1)]
    pts = points[~np.isinf((points)).any(axis=1)]
    pts = np.divide(pts,1000.0)
    v =pptk.viewer(pts,rgb)
    v.set(point_size=0.1)
    v.wait()

    # mask = disp > disp.min()
    # out_points = points[mask]
    # out_colors = colors[mask]
    out_fn = 'out.ply'

    # write_ply('out.ply', out_points, out_colors)

    print('%s saved' % 'out.ply')
    num_disp = 144

    # cv2.imshow('left', imgL_BGR)
    # cv2.imshow('disparity', (disp-min_disp)/num_disp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    mask = disp > disp.min()

    # Mask colors and points.

    output_points = points[mask]

    output_colors = colors[mask]

    # Define name for output file

    output_file = 'reconstructed.ply'

    # Generate point cloud

    print("\n Creating the output file... \n")

    create_output(output_points, output_colors, output_file)