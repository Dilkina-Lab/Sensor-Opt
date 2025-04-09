import matplotlib.pyplot as plt

def visualize_activity_centers(raster, points, display=True):
    # imshow [M,N] puts (0,0) top left, (M,0) bottom left, (0,N) top right
    plt.imshow(raster)

    # origin is top left
    # scatter seems to plot 1st coord as horizontal axis, 2nd coord as vertical
    # so use horizontal axis to specify col (2nd coord in points), vertical axis for row (1st coord in points)
    points_x = [p[1] for p in points]
    points_y = [p[0] for p in points]
    
    # plt coordinate (0, 0) is at the center of raster grid cell (0, 0)
    points_x_plot = [p - 0.5 for p in points_x]
    points_y_plot = [p - 0.5 for p in points_y]
    plt.scatter(points_x_plot, points_y_plot, c='red', s=10)

    cbar = plt.colorbar()
    cbar.set_label('$z(x,y)$')
    
    # cut off image at raster boundaries
    plt.gca().axis([-0.5, raster.shape[0]-0.5, raster.shape[1]-0.5, -0.5])
    
    if display:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig('activity_centers.png')
        plt.close()

def visualize_trap_layout(raster, grid_loc, display=True):
    # imshow [M,N] puts (0,0) top left, (M,0) bottom left, (0,N) top right
    plt.imshow(raster)

    # origin is top left
    # scatter seems to plot 1st coord as horizontal axis, 2nd coord as vertical
    # so use horizontal axis to specify col (2nd coord in points), vertical axis for row (1st coord in points)
    # plt coordinate (0, 0) is at the center of raster grid cell (0, 0)
    grid_x_plot = [p[1] - 0.5 for p in grid_loc]
    grid_y_plot = [p[0] - 0.5 for p in grid_loc]
    plt.scatter(grid_x_plot, grid_y_plot, c='black', marker='+', s=40)

    cbar = plt.colorbar()
    cbar.set_label('$z(x,y)$')

    if display:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig('trap_locations.png')
        plt.close()

def visualize_activity_centers_and_trap_layout(raster, points, grid_loc, display=True):
    # imshow [M,N] puts (0,0) top left, (M,0) bottom left, (0,N) top right
    plt.imshow(raster)

    # origin is top left
    # scatter seems to plot 1st coord as horizontal axis, 2nd coord as vertical
    # so use horizontal axis to specify col (2nd coord in points), vertical axis for row (1st coord in points)
    points_x = [p[1] for p in points]
    points_y = [p[0] for p in points]
    
    # plt coordinate (0, 0) is at the center of raster grid cell (0, 0)
    points_x_plot = [p - 0.5 for p in points_x]
    points_y_plot = [p - 0.5 for p in points_y]
    plt.scatter(points_x_plot, points_y_plot, c='red', s=10)

    grid_x_plot = [p[1] - 0.5 for p in grid_loc]
    grid_y_plot = [p[0] - 0.5 for p in grid_loc]
    plt.scatter(grid_x_plot, grid_y_plot, c='black', marker='+', s=40)

    cbar = plt.colorbar()
    cbar.set_label('$z(x,y)$')

    # cut off image at raster boundaries
    plt.gca().axis([-0.5, raster.shape[0]-0.5, raster.shape[1]-0.5, -0.5])


    if display:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig('activity_centers_and_trap_locations.png')
        plt.close()