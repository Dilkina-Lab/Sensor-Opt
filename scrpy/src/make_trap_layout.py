import argparse
import csv
import itertools
import numpy as np
import rasterio
import time

def create_grid_layout(n, m, raster, buffer=0.1):
    n_pix_h, n_pix_w = raster.shape
    grid_xmin = n_pix_h*buffer
    grid_xmax = n_pix_h*(1-buffer)
    grid_ymin = n_pix_w*buffer
    grid_ymax = n_pix_w*(1-buffer)

    grid_x = np.linspace(grid_xmin, grid_xmax, n)
    grid_y = np.linspace(grid_ymin, grid_ymax, m)
    grid_loc = list(itertools.product(grid_x, grid_y))

    return grid_loc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--landscape_raster_file', required=True, type=str, help='Landscape for which to generate trap layout')
    parser.add_argument('--trap_config', required=True, type=str, help='Configuration for trap layout.')
    parser.add_argument('--n_traps_x', type=int, help='Number of traps to place along x direction of grid configuration.')
    parser.add_argument('--n_traps_y', type=int, help='Number of traps to place along y direction of grid configuration.')
    args = parser.parse_args()

    # Load landscape
    landscape_raster = rasterio.open(args.landscape_raster_file)
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    landscape_raster_name = args.landscape_raster_file.split('/')[-1].split('.tif')[0]

    # Simulate grid of trap locations
    if args.trap_config == 'grid':
        timer = time.time()
        trap_loc = create_grid_layout(args.n_traps_x, args.n_traps_y, landscape_ndarr)
        print('Simulated trap locations in %0.2f seconds'%(time.time() - timer))
        with open('../data/simulation/trap_locations/'+landscape_raster_name+'_grid_'+str(args.n_traps_x)+'_'+str(args.n_traps_y)+'.csv', 'w') as f:
            tlwriter = csv.writer(f)
            for p in trap_loc:
                tlwriter.writerow([p[0], p[1]])

if __name__ == '__main__':
    main()