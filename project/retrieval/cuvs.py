import os
from datetime import datetime

from myeachtra.dependencies import memory
import cupy as cp
import numpy as np
import rmm
from pylibraft.neighbors import cagra, ivf_flat

from pylibraft.common import DeviceResources
from rmm.allocators.cupy import rmm_cupy_allocator

from query_parse.visual import norm_photo_features, photo_ids

mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=2 ** 15)
rmm.mr.set_current_device_resource(mr)
cp.cuda.set_allocator(rmm_cupy_allocator)
handle = DeviceResources()

# Load the photo features and image IDs
vectors = cp.asarray(norm_photo_features, dtype=np.float32)

ALGO = "ivf_flat"
if ALGO == "ivf_flat":
    algo = ivf_flat
    index_params = algo.IndexParams(
        n_lists=2048,  # Number of inverted lists
        metric="inner_product",  # Distance metric
        add_data_on_build=True,  # Add the data to the index when building
        kmeans_n_iters=25,  # Number of iterations for k-means
    )
    search_params = algo.SearchParams(n_probes=64)
else:
    algo = cagra
    index_params = algo.IndexParams(
        metric="inner_product",  # Distance metric
    )
    search_params = algo.SearchParams(
        itopk_size=64,
        algo="multi_cta",
        team_size=4,
    )

# Build the index (this step is necessary before performing searches)
rebuild = False
path = os.path.join("cachedir", ALGO + ".index")
if os.path.exists(path) and not rebuild:
    index = algo.load(path, handle=handle)
    handle.sync()
else:
    print("Building RAFT index...")
    start_time = datetime.now()
    index = algo.build(index_params, vectors, handle=handle)
    handle.sync()
    print("Time taken:", (datetime.now() - start_time).total_seconds(), "seconds")
    print("Saving RAFT index...")
    algo.save(path, index)

# Function to perform a search
@memory.cache
def search(query_vector, top_k=5):
    query_vector = cp.asarray(query_vector, dtype=np.float32).reshape(1, -1)



    # Perform the search
    handle = DeviceResources()
    distances, indices = algo.search(search_params, index, query_vector, k=top_k, handle=handle)
    handle.sync()
    # Convert the indices to image IDs
    indices = cp.asnumpy(indices).flatten()
    images = np.array(photo_ids)[indices]
    distances = cp.asnumpy(distances).reshape(-1)
    return {image: distance for image, distance in zip(images, distances)}, images
