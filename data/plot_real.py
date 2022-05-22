import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

string = "real_stage1"
act_graph = np.load("action_"+string+".npy",allow_pickle=True)
pessi_graph = np.load("pessi_"+string+".npy",allow_pickle=True)
non_pessi_graph = np.load("non_pessi_"+string+".npy",allow_pickle=True)
uniform_graph = np.load("uniform_"+string+".npy",allow_pickle=True)

x_edges = np.arange(-0.5,5)
y_edges = np.arange(-0.5,5)
f, ax = plt.subplots(1, 3, figsize=(12,3))
ax1 = ax[0]
ax1.imshow(np.flipud(act_graph), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
ax2 = ax[1]
ax2.imshow(np.flipud(non_pessi_graph), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
ax5 = ax[2]
ax5.imshow(np.flipud(uniform_graph), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])

# Major ticks
ax1.set_xticks(np.arange(0, 5, 1));
ax1.set_yticks(np.arange(0, 5, 1));
ax2.set_xticks(np.arange(0, 5, 1));
ax2.set_yticks(np.arange(0, 5, 1));
ax5.set_xticks(np.arange(0, 5, 1));
ax5.set_yticks(np.arange(0, 5, 1));

# Labels for major ticks
ax1.set_xticklabels(np.arange(0, 5, 1));
ax1.set_yticklabels(np.arange(0, 5, 1));
ax2.set_xticklabels(np.arange(0, 5, 1));
ax2.set_yticklabels(np.arange(0, 5, 1));
ax5.set_xticklabels(np.arange(0, 5, 1));
ax5.set_yticklabels(np.arange(0, 5, 1));

# Minor ticks
ax1.set_xticks(np.arange(-.5, 5, 1), minor=True);
ax1.set_yticks(np.arange(-.5, 5, 1), minor=True);
ax2.set_xticks(np.arange(-.5, 5, 1), minor=True);
ax2.set_yticks(np.arange(-.5, 5, 1), minor=True);
ax5.set_xticks(np.arange(-.5, 5, 1), minor=True);
ax5.set_yticks(np.arange(-.5, 5, 1), minor=True);

# Gridlines based on minor ticks
ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
ax2.grid(which='minor', color='b', linestyle='-', linewidth=1)
ax5.grid(which='minor', color='b', linestyle='-', linewidth=1)

im1 = ax1.pcolormesh(x_edges, y_edges, act_graph, cmap='Blues', vmin=0, vmax=5000)
f.colorbar(im1, ax=ax1, label = "Action counts")
im2 = ax2.pcolormesh(x_edges, y_edges, non_pessi_graph, cmap='Blues', vmin=0, vmax=5000)
f.colorbar(im2, ax=ax2, label = "Action counts")
im5 = ax5.pcolormesh(x_edges, y_edges, uniform_graph, cmap='Blues', vmin=0, vmax=5000)
f.colorbar(im5, ax=ax5, label = "Action counts")

ax1.set_ylabel('IV Fluid', fontsize=12)
ax1.set_xlabel('VP', fontsize=12)
ax2.set_ylabel('IV Fluid', fontsize=12)
ax2.set_xlabel('VP', fontsize=12)
ax5.set_ylabel('IV Fluid', fontsize=12)
ax5.set_xlabel('VP', fontsize=12)

ax1.set_title("Physician", fontsize=12)
ax2.set_title("Non-Pessi", fontsize=12)
ax5.set_title("PBL", fontsize=12)

plt.tight_layout()
f.savefig('../stage1_real.png')


string = "real_stage2"
act_graph1 = np.load("action1_"+string+".npy",allow_pickle=True)
pessi_graph1 = np.load("pessi1_"+string+".npy",allow_pickle=True)
non_pessi_graph1 = np.load("non_pessi1_"+string+".npy",allow_pickle=True)
uniform_graph1 = np.load("uniform1_"+string+".npy",allow_pickle=True)
act_graph2 = np.load("action2_"+string+".npy",allow_pickle=True)
pessi_graph2 = np.load("pessi2_"+string+".npy",allow_pickle=True)
non_pessi_graph2 = np.load("non_pessi2_"+string+".npy",allow_pickle=True)
uniform_graph2 = np.load("uniform2_"+string+".npy",allow_pickle=True)

act_graph = act_graph1 + act_graph2
pessi_graph = pessi_graph1 + pessi_graph2
non_pessi_graph = non_pessi_graph1 + non_pessi_graph2
uniform_graph = uniform_graph1 + uniform_graph2

x_edges = np.arange(-0.5,5)
y_edges = np.arange(-0.5,5)
f, ax = plt.subplots(1, 3, figsize=(12,3))
ax1 = ax[0]
ax1.imshow(np.flipud(act_graph), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
ax2 = ax[1]
ax2.imshow(np.flipud(non_pessi_graph), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
ax5 = ax[2]
ax5.imshow(np.flipud(uniform_graph), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])

# Major ticks
ax1.set_xticks(np.arange(0, 5, 1));
ax1.set_yticks(np.arange(0, 5, 1));
ax2.set_xticks(np.arange(0, 5, 1));
ax2.set_yticks(np.arange(0, 5, 1));
ax5.set_xticks(np.arange(0, 5, 1));
ax5.set_yticks(np.arange(0, 5, 1));

# Labels for major ticks
ax1.set_xticklabels(np.arange(0, 5, 1));
ax1.set_yticklabels(np.arange(0, 5, 1));
ax2.set_xticklabels(np.arange(0, 5, 1));
ax2.set_yticklabels(np.arange(0, 5, 1));
ax5.set_xticklabels(np.arange(0, 5, 1));
ax5.set_yticklabels(np.arange(0, 5, 1));

# Minor ticks
ax1.set_xticks(np.arange(-.5, 5, 1), minor=True);
ax1.set_yticks(np.arange(-.5, 5, 1), minor=True);
ax2.set_xticks(np.arange(-.5, 5, 1), minor=True);
ax2.set_yticks(np.arange(-.5, 5, 1), minor=True);
ax5.set_xticks(np.arange(-.5, 5, 1), minor=True);
ax5.set_yticks(np.arange(-.5, 5, 1), minor=True);

# Gridlines based on minor ticks
ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
ax2.grid(which='minor', color='b', linestyle='-', linewidth=1)
ax5.grid(which='minor', color='b', linestyle='-', linewidth=1)

im1 = ax1.pcolormesh(x_edges, y_edges, act_graph, cmap='Blues', vmin=0, vmax=5000)
f.colorbar(im1, ax=ax1, label = "Action counts")
im2 = ax2.pcolormesh(x_edges, y_edges, non_pessi_graph, cmap='Blues', vmin=0, vmax=5000)
f.colorbar(im2, ax=ax2, label = "Action counts")
im5 = ax5.pcolormesh(x_edges, y_edges, uniform_graph, cmap='Blues', vmin=0, vmax=5000)
f.colorbar(im5, ax=ax5, label = "Action counts")

ax1.set_ylabel('IV Fluid', fontsize=12)
ax1.set_xlabel('VP', fontsize=12)
ax2.set_ylabel('IV Fluid', fontsize=12)
ax2.set_xlabel('VP', fontsize=12)
ax5.set_ylabel('IV Fluid', fontsize=12)
ax5.set_xlabel('VP', fontsize=12)

ax1.set_title("Physician", fontsize=12)
ax2.set_title("Non-Pessi", fontsize=12)
ax5.set_title("PBL", fontsize=12)

plt.tight_layout()
f.savefig('../stage2_real.png')