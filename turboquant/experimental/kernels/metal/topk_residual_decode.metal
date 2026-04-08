    uint gid = thread_position_in_grid.x;
    uint d_g = N_GROUPS * GROUP_SIZE;

    uint prefix_idx = gid / d_g;
    uint local_idx_in_row = gid % d_g;
    uint group_idx = local_idx_in_row / GROUP_SIZE;
    uint local_idx = local_idx_in_row % GROUP_SIZE;

    float val = 0.0f;
    uint base_idx = (prefix_idx * N_GROUPS + group_idx) * TOPK;

    #pragma unroll
    for (uint i = 0; i < TOPK; ++i) {
        bool match = (uint(indices[base_idx + i]) == local_idx);
        val += match ? (float)values[base_idx + i] : 0.0f;
    }

    out[gid] = typename decltype(out)::value_type(val);
