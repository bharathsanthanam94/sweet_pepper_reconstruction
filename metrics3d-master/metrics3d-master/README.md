# Metrics3D

Collection of commonly used 3d metrics for python.<br />
For any doubts contact Federico.

## Installation

`python3 install.py`

Note that the depencies are not installed by default.\
However, we only rely on standard libraries such as `numpy`, `scipy`, `open3d` and `matplotlib`

## How to use Metrics3D

Here's a snippet to use 'Metrics3D' in your implementations

```
from metrics_3d.chamfer_distance import ChamferDistance

cd = ChamferDistance()

cd.update(o3d.mesh, o3d.mesh)  
cd.update(o3d.mesh, o3d.pcd)  
cd.update(o3d.pcd, o3d.mesh)  
cd.update(o3d.pcd, o3d.pcd)  

metric = cd.compute()

cd.reset()

```
