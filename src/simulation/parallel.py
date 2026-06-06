"""
parallel.py - Custom MPI layer that distributes the 448 bunches over a small
number of ranks (28 bunches/rank for 16 ranks).

mbtrack2's built-in Beam(..., mpi=True) maps one bunch to one rank, which would
require 448 ranks for a uniformly filled ALBA II ring. These helpers instead
distribute the bunches contiguously and exchange the longitudinal profiles and
per-bunch statistics across all ranks every turn.
"""

import numpy as np

from constants import H
from config import CAVITY_NBIN


def validate_parallel_layout(comm):
    """
    Checks that the number of MPI ranks divides the harmonic number H exactly,
    so that bunches can be distributed evenly across ranks. Raises ValueError
    otherwise. Returns (size, bunches_per_rank).
    """

    size = comm.Get_size()
    if H % size != 0:
        raise ValueError(
            f"This script requires the MPI size to divide H exactly. Got H={H} and size={size}."
        )
    return size, H // size


def local_bunch_indices(rank: int, size: int):
    """
    Returns the array of bunch indices assigned to this MPI rank.
    Bunches are distributed contiguously: rank 0 owns [0, ..., H/size - 1],
    rank 1 owns [H/size, ..., 2*H/size - 1], and so on.
    """

    bunches_per_rank = H // size
    start = rank * bunches_per_rank
    stop = start + bunches_per_rank
    return np.arange(start, stop, dtype=int)


def compute_local_profiles(local_indices_arr, local_bunches):
    """
    Computes the longitudinal bin profiles for all bunches assigned to this MPI
    rank. For each bunch, calls mbtrack2's binning() method in the tau dimension
    to produce a histogram of the macro-particle distribution. Returns the bin
    centres, profiles (particle counts), bin widths, charge per macro-particle,
    and sorted particle indices, all needed by the cavity kick routines on the
    next step.
    """

    n_local = len(local_indices_arr)
    # np.empty is safe here: the loop below fills every slot before use.
    local_centers = np.empty((n_local, CAVITY_NBIN), dtype=np.float64)
    local_profiles = np.empty((n_local, CAVITY_NBIN), dtype=np.int64)
    local_bin_lengths = np.empty(n_local, dtype=np.float64)
    local_charge_per_mp = np.empty(n_local, dtype=np.float64)
    local_sorted_index = []

    for slot, bunch_idx in enumerate(local_indices_arr):
        bunch = local_bunches[bunch_idx]

        # bunch.binning() histograms the macro-particle distribution along one phase-
        # space dimension into n_bin intervals. Returns bin edges, particle indices
        # sorted by position, per-bin particle counts, and bin centre positions.
        bins, sorted_index, profile, center = bunch.binning(dimension="tau",
                                                            n_bin=CAVITY_NBIN)
        local_centers[slot, :] = center
        local_profiles[slot, :] = profile
        local_bin_lengths[slot] = bins[1] - bins[0]
        local_charge_per_mp[slot] = bunch.charge_per_mp
        local_sorted_index.append(sorted_index)

    return local_centers, local_profiles, local_bin_lengths, local_charge_per_mp, local_sorted_index


def allgather_profiles(comm, local_centers, local_profiles, local_bin_lengths,
                       local_charge_per_mp):
    """
    MPI communication step for the longitudinal bunch profiles. This function
    takes the longitudinal profiles computed locally on each MPI rank and makes
    every rank receive a complete copy of the profiles of all bunches in the ring.
    This is necessary because the cavity beam loading depends on the passage of
    the full bunch train, not only on the bunches owned by a given rank. In
    run_tracking(), this function is called immediately after
    compute_local_profiles() and immediately before track_cavity_distributed().

    Each rank contributes its own n_local bunches and receives the full train
    (H bunches) in return. The gathered arrays are first stored with shape
        (size, n_local, CAVITY_NBIN)
    where size is the number of MPI ranks. They are then reshaped to
        (H, CAVITY_NBIN)
    so that the cavity tracker can iterate over all RF buckets in their global order.
    """

    size = comm.Get_size()
    n_local = local_centers.shape[0]

    global_centers = np.empty((size, n_local, CAVITY_NBIN), dtype=np.float64)
    global_profiles = np.empty((size, n_local, CAVITY_NBIN), dtype=np.int64)
    global_bin_lengths = np.empty((size, n_local), dtype=np.float64)
    global_charge_per_mp = np.empty((size, n_local), dtype=np.float64)


    # How the communication works: each rank contributes its n_local profiles
    # (shape (n_local, NBIN)); comm.Allgather stacks all contributions into a
    # (size, n_local, NBIN) array and delivers an identical full copy to every
    # rank. The reshape from (size, n_local, NBIN) to (H, NBIN) merges the rank and
    # per-rank-bunch axes into a single bunch axis (size * n_local = H).
    # Because bunches are distributed contiguously, this preserves the global
    # bucket order. After this call every rank holds all H profiles, ready for
    # the cavity tracker.
    # No cross-bunch arithmetic happens here: the inter-bunch coupling (the
    # cavity wake one bunch leaves for the next) is applied later, bucket by
    # bucket, inside track_cavity_distributed.
    comm.Allgather(local_centers, global_centers)
    comm.Allgather(local_profiles, global_profiles)
    comm.Allgather(local_bin_lengths, global_bin_lengths)
    comm.Allgather(local_charge_per_mp, global_charge_per_mp)

    return (
        global_centers.reshape(H, CAVITY_NBIN),
        global_profiles.reshape(H, CAVITY_NBIN),
        global_bin_lengths.reshape(H),
        global_charge_per_mp.reshape(H),
    )


def gather_checkpoint_stats(comm, local_indices_arr, local_bunches):
    """
    Collects per-bunch longitudinal statistics from all MPI ranks into rank 0.
    Also gathers sigma_delta, eps_x, eps_y for the ELEGANT-equivalent plots.
    Dispersion correction not needed because DISPERSION_LOCAL = [0, 0, 0, 0].
    """
    n_local = len(local_indices_arr)
    local_sigma_t = np.empty(n_local, dtype=np.float64)
    local_mean_tau = np.empty(n_local, dtype=np.float64)
    local_mean_delta = np.empty(n_local, dtype=np.float64)
    local_sigma_delta = np.empty(n_local, dtype=np.float64)
    local_eps_x      = np.empty(n_local, dtype=np.float64)
    local_eps_y      = np.empty(n_local, dtype=np.float64)

    for slot, bunch_idx in enumerate(local_indices_arr):
        bunch = local_bunches[bunch_idx]

        # mbtrack2 phase-space order: [x, xp, y, yp, tau, delta] → indices 4, 5.
        local_sigma_t[slot] = bunch.std[4]
        local_mean_tau[slot] = bunch.mean[4]
        local_mean_delta[slot] = bunch.mean[5]
        local_sigma_delta[slot] = bunch.std[5]
        x, xp = bunch["x"], bunch["xp"]
        y, yp = bunch["y"], bunch["yp"]
        xc, xpc = x - x.mean(), xp - xp.mean()
        yc, ypc = y - y.mean(), yp - yp.mean()
        local_eps_x[slot] = np.sqrt(max(
            np.mean(xc*xc)*np.mean(xpc*xpc) - np.mean(xc*xpc)**2, 0.0))
        local_eps_y[slot] = np.sqrt(max(
            np.mean(yc*yc)*np.mean(ypc*ypc) - np.mean(yc*ypc)**2, 0.0))

    if comm.rank == 0:
        global_sigma_t = np.empty(H, dtype=np.float64)
        global_mean_tau = np.empty(H, dtype=np.float64)
        global_mean_delta = np.empty(H, dtype=np.float64)
        global_sigma_delta = np.empty(H, dtype=np.float64)
        global_eps_x       = np.empty(H, dtype=np.float64)
        global_eps_y       = np.empty(H, dtype=np.float64)
    else:
        global_sigma_t = global_mean_tau = global_mean_delta = None
        global_sigma_delta = global_eps_x = global_eps_y = None

    comm.Gather(local_sigma_t, global_sigma_t, root=0)
    comm.Gather(local_mean_tau, global_mean_tau, root=0)
    comm.Gather(local_mean_delta, global_mean_delta, root=0)
    comm.Gather(local_sigma_delta, global_sigma_delta, root=0)
    comm.Gather(local_eps_x, global_eps_x, root=0)
    comm.Gather(local_eps_y, global_eps_y, root=0)


    return (global_sigma_t, global_mean_tau, global_mean_delta,
            global_sigma_delta, global_eps_x, global_eps_y)
