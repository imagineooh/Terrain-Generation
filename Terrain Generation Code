from random import uniform, randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


octaves=20
arrays = {f"array_{i}": np.zeros(3) for i in range(octaves)}
def make_charts_n(size):

    for s in range(octaves):
        np.set_printoptions(precision=6, suppress=True)
        noise_table=np.empty((size,size))
        noise_table[0, 0] = uniform(-1, 1)
        for i in range(size):
            for j in range(size):


                if i==1 and j==1:
                    continue

                if i ==0:
                    noise_table[i,j]= noise_table[i, j-1]+ uniform(-1,1)


                elif j==0:
                    noise_table[i,j]= noise_table[i-1,j]+  uniform(-1,1)

                elif j>0 and i>0:
                    noise_table[i,j]= (noise_table[i-1,j]+noise_table[i,j-1]+noise_table[i-1,j-1])/3 + uniform(-1,1)


                else:
                    noise_table[i,j]=noise_table[i-1,j]+ uniform(-1,1)


        rotated_fig= np.rot90(noise_table)



        prefinal_array= rotated_fig+noise_table
        rotated_fig = np.rot90(prefinal_array)

        arrays[f"array_{s}"] = prefinal_array

make_charts_n(160)
sample_array = arrays["array_0"]
y, x = sample_array.shape


total_sum = sum(arrays.values())/octaves

smoothed_data = gaussian_filter(total_sum, sigma=10)

low_res_noise = np.empty((x // 4, y // 4))
height, width = low_res_noise.shape

for i in range(height):
    for j in range(width):
        if i == 0 and j == 0:
            low_res_noise[i, j] = uniform(-1, 1)
        elif i == 0:
            low_res_noise[i, j] = low_res_noise[i, j - 1] + uniform(-1, 1)
        elif j == 0:
            low_res_noise[i, j] = low_res_noise[i - 1, j] + uniform(-1, 1)
        else:
            low_res_noise[i, j] = (
                (low_res_noise[i - 1, j] + low_res_noise[i, j - 1] + low_res_noise[i - 1, j - 1]) / 3
                + uniform(-1, 1)
            )
low_res_noise=np.kron(low_res_noise, np.ones((4, 4)))
low_res_noise= gaussian_filter(low_res_noise, sigma=10)

blended_map= smoothed_data+ 0.3*low_res_noise
blended_map=np.absolute(blended_map)-(np.max(blended_map))/2



#initializing the water map
"""in this section, the algorithm is evaluating how it should generate the water. contact me for further inquiry"""


water_octaves=10
water_arrays = {f"array_{i}": np.zeros(3) for i in range(water_octaves)}

def make_water_charts_n(size):

    for s in range(water_octaves):
        np.set_printoptions(precision=6, suppress=True)
        water_table=np.empty((size,size))
        water_table[0, 0] = uniform(-1, 1)
        for i in range(size):
            for j in range(size):


                if i==1 and j==1:
                    continue

                if i ==0:
                    water_table[i,j]= water_table[i, j-1]+ uniform(-1,1)


                elif j==0:
                    water_table[i,j]= water_table[i-1,j]+  uniform(-1,1)

                elif j>0 and i>0:
                    water_table[i,j]= (water_table[i-1,j]+water_table[i,j-1]+water_table[i-1,j-1])/3 + uniform(-1,1)


                else:
                    water_table[i,j]=water_table[i-1,j]+ uniform(-1,1)


        rotated_fig= np.rot90(water_table)



        water_prefinal_array= rotated_fig+water_table
        rotated_fig = np.rot90(water_prefinal_array)

        water_arrays[f"array_{s}"] = water_prefinal_array

make_water_charts_n(160)
water_arrays_smoothed = {
    key: gaussian_filter(array, sigma=20)
    for key, array in water_arrays.items()
}

total_water_sum = sum(water_arrays_smoothed.values()) / water_octaves
lake_water_level = np.max(total_water_sum) * 0.8  # This is the lake water level, can be adjusted




lake_map = total_water_sum > np.max(total_water_sum) * 0.9# Value between 0 - 1. This value affects the lake visibility

lake_map= gaussian_filter(lake_map, sigma=5)


blended_water_terrain = blended_map - lake_map
sediment_take=0
changed = True
starting_points = [(100, 100), (120, 120), (140, 140)]  # Example points
# Erosion process based on starting points
highlight_coords = []  # Coordinates of valleys (sediment-deposited points)
highlight_values = []  # Values at those coordinates

starting_points = [(randint(0, 160), randint(0, 160)),
                  (randint(0, 160), randint(0, 160)),
                  (randint(0, 160), randint(0, 160)),
                  (randint(0, 160), randint(0, 160))]

changed = True
while changed:
    changed = False
    for _ in range(500):
        for start in starting_points:
            x, y = start

            neighbors = [
                (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
            ]

            for nx, ny in neighbors:
                if 0 <= nx < 159 and 0 <= ny < 160:
                    if blended_water_terrain[nx, ny] < blended_water_terrain[x, y]:
                        diff = (blended_water_terrain[x, y] - blended_water_terrain[nx, ny]) // 2
                        if diff > 0:
                            blended_water_terrain[x, y] -= diff
                            blended_water_terrain[x,y]=blended_water_terrain[x,y]-1
                            blended_water_terrain[nx, ny] += diff * 0.1  
                            highlight_coords.append((nx, ny))  
                            highlight_values.append(blended_water_terrain[nx, ny])
                            changed = True
                    else:
                        blended_water_terrain[x, y] += sediment_take / 2

# Plot the valleys (sediment-deposited points) in blue
for (x, y), value in zip(highlight_coords, highlight_values):
    plt.scatter(y, x, color='blue', s=100, edgecolor='black')

# Optional: Plot points above threshold in red
threshold = 5
coords_above_threshold = np.argwhere(total_water_sum > threshold)

for (x, y) in coords_above_threshold:
    plt.scatter(y, x, color='red', s=100, edgecolor='black')



"""
for coord in coords_above_threshold:
    x, y = coord  
    blended_water_terrain[x, y] -= 10  

for coord in coords_above_threshold:
    x, y = coord  

    hole_depth = -blended_water_terrain[x, y]
    water_height = min(hole_depth, 5)


    # Add the water at the hole
    blended_water_terrain[x, y] = max(blended_water_terrain[x, y], lake_water_level + water_height)
"""
#blended_water_terrain=blended_water_terrain-lake_map

#blended_water_terrain=gaussian_filter(blended_water_terrain, sigma=1)
water_level = (np.max(blended_map)+np.min(blended_map))/2 #I chose not to put zero so that you can see as much over the water than under


terrain_with_water = np.copy(blended_water_terrain)


terrain_with_water[terrain_with_water < water_level] = water_level


x = np.arange(blended_water_terrain.shape[0])
y = np.arange(blended_water_terrain.shape[1])
x, y = np.meshgrid(x, y, indexing="ij")


water_plane = np.full_like(blended_water_terrain, water_level)


fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(121, projection='3d')


ax1.plot_surface(x, y, blended_water_terrain, cmap='terrain', alpha=1)


ax1.plot_surface(x, y, water_plane, color='blue', alpha=0.5)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Height')
ax1.set_title('Blended Water Terrain - 3D')


ax2 = fig.add_subplot(122)
ax2.imshow(terrain_with_water, cmap='terrain', interpolation='nearest')  
ax2.set_title('Blended Water Terrain - 2D')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.show()
