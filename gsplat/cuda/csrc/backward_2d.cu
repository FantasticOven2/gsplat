#include "backward_2d.cuh"
#include "helpers.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile) {
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile) {
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile) {
    val = cg::reduce(tile, val, cg::plus<float>());
}


__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float2* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float* __restrict__ transMats,
    const float3* __restrict__ u_transforms,
    const float3* __restrict__ v_transforms,
    const float3* __restrict__ w_transforms,

    // grad input 
    float* __restrict__ dL_dtransMats,
    float3* __restrict__ v_u_transforms,
    float3* __restrict__ v_v_transforms,
    float3* __restrict__ v_w_transforms,
    // const float* __restrict__ dL_dnormal3Ds,

    // grad output
    float3* __restrict__ v_mean3Ds,
    float2* __restrict__ v_scales,
    float4* __restrict__ v_quats,
    float3* __restrict__ v_mean2Ds

) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) return;


    const float* transMat = &(transMats[9 * idx]);
    const float* dL_dtransMat = &(dL_dtransMats[9 * idx]);
    // const float* dL_dnormal3D = &(dL_dnormal3Ds[3 * idx]);

    glm::vec3 p_world = glm::vec3(means3d[idx].x, means3d[idx].y, means3d[idx].z);
    float fx = intrins.x;
    float fy = intrins.y;
    float tan_fovx = intrins.z / fx;
    float tan_fovy = intrins.w / fy;
    
    glm::vec3 v_mean3D;
    glm::vec2 v_scale;
    glm::vec4 v_quat;

    build_AABB(
        num_points,
        radii,
        img_size.x,
        img_size.y,
        transMats,
        u_transforms,
        v_transforms,
        w_transforms,

        v_mean2Ds,
        dL_dtransMats,
        v_u_transforms,
        v_v_transforms,
        v_w_transforms
    );
    
    build_H(
        p_world,
        quats[idx],
        scales[idx],
        viewmat,
        intrins,
        tan_fovx,
        tan_fovy,
        transMat,
        u_transforms[idx],
        v_transforms[idx],
        w_transforms[idx],

        // grad input
        dL_dtransMat,
        v_u_transforms[idx],
        v_v_transforms[idx],
        v_w_transforms[idx],
        // dL_dnormal3D,

        // grad output
        v_mean3D,
        v_scale,
        v_quat
    );

    // Update 
    float3 v_mean3D_float = {v_mean3D.x, v_mean3D.y, v_mean3D.z};
    float2 v_scale_float = {v_scale.x, v_scale.y};
    float4 v_quat_float = {v_quat.x, v_quat.y, v_quat.z, v_quat.w};
    
    v_mean3Ds[idx] = v_mean3D_float;
    v_scales[idx] = v_scale_float;
    v_quats[idx] = v_quat_float;
}


// For simplest 2DGS normal constraint is not used, therefore exclude it for now
__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float* __restrict__ transMats,
    const float3* __restrict__ u_transforms,
    const float3* __restrict__ v_transforms,
    const float3* __restrict__ w_transforms,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,

    // grad input
    const float3* __restrict__ dL_doutput,
    const float* __restrict__ dL_doutput_alpha,

    // grad_output
    float2* __restrict__ v_mean2D,
    float* __restrict__ dL_dtransMat,
    float3* __restrict__ v_u_transforms,
    float3* __restrict__ v_v_transforms,
    float3* __restrict__ v_w_transforms,
    float3* __restrict__ dL_drgb,
    float* __restrict__ dL_dopacity
) {
    // printf("It is real :) \n");
    auto block = cg::this_thread_block();
    int32_t tile_id = 
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i = 
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = 
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside ? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tu_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tv_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tw_batch[MAX_BLOCK_SIZE];

    __shared__ float3 u_transform_batch[MAX_BLOCK_SIZE];
    __shared__ float3 v_transform_batch[MAX_BLOCK_SIZE];
    __shared__ float3 w_transform_batch[MAX_BLOCK_SIZE];
    

    // df/d_out for this pixel
    const float3 v_out = dL_doutput[pix_id];
    const float v_out_alpha = dL_doutput_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            rgbs_batch[tr] = rgbs[g_id];

            //====== 2D Splatting ======//
            Tu_batch[tr] = {transMats[9 * g_id + 0], transMats[9 * g_id + 1], transMats[9 * g_id + 2]};
            Tv_batch[tr] = {transMats[9 * g_id + 3], transMats[9 * g_id + 4], transMats[9 * g_id + 5]};
            Tw_batch[tr] = {transMats[9 * g_id + 6], transMats[9 * g_id + 7], transMats[9 * g_id + 8]};

            //====== New Version ======//
            u_transform_batch[tr] = u_transforms[g_id];
            v_transform_batch[tr] = v_transforms[g_id];
            w_transform_batch[tr] = w_transforms[g_id];

        }
        // wait for other threads to collext the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float vis;
            float rho3d;
            float rho2d;
            float rho;
            float2 s;
            float2 d;
            float3 k;
            float3 l;
            float3 p;
            float3 Tw;
            float3 w_transform;
            if (valid) {
                // Here we compute gaussian weights in forward pass again
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                const float2 xy = {xy_opac.x, xy_opac.y};
                const float3 Tu = Tu_batch[t];
                const float3 Tv = Tv_batch[t];
                Tw = Tw_batch[t];
                
                const float3 u_transform = u_transform_batch[t];
                const float3 v_transform = v_transform_batch[t];
                w_transform = w_transform_batch[t];

                // k = {-Tu.x + px * Tw.x, -Tu.y + px * Tw.y, -Tu.z + px * Tw.z};
                // l = {-Tv.x + py * Tw.x, -Tv.y + py * Tw.y, -Tv.z + py * Tw.z};
                
                k = {-u_transform.x + px * w_transform.x, -u_transform.y + px * w_transform.y, -u_transform.z + px * w_transform.z};
                l = {-v_transform.x + py * w_transform.x, -v_transform.y + py * w_transform.y, -v_transform.z + py * w_transform.z};

                p = cross_product(k, l);
                
                if (p.z == 0.0) {
                   continue;
                } else {
                    s = {p.x / p.z, p.y / p.z};
                }
                rho3d = (s.x * s.x + s.y * s.y);
                d = {xy.x - px, xy.y - py};
                rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

                // compute intersection
                rho = min(rho3d, rho2d);

                // accumulations
                float sigma = 0.5f * rho;

                vis = __expf(-sigma);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) continue;

            float3 dL_drgb_local = {0.f, 0.f, 0.f};
            float dL_dopacity_local = 0.f;
            float2 v_mean2D_local = {0.f, 0.f};
            float3 dL_dTu_local = {0.f, 0.f, 0.f};
            float3 dL_dTv_local = {0.f, 0.f, 0.f};
            float3 dL_dTw_local = {0.f, 0.f, 0.f};
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update dL_drgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                dL_drgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                // Helpful reusable temporary variable
                const float dL_dG = opac * v_alpha;
                float dL_dz = 0.0f;

                int32_t g = id_batch[t];

                // ====== 2D Splatting ====== //
                if (rho3d <= rho2d) {
                    // printf("dL_dG: %.2f \n", dL_dG);
                    // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
                    const float2 dL_ds = {
                        dL_dG * -vis * s.x + dL_dz * Tw.x,
                        dL_dG * -vis * s.y + dL_dz * Tw.y
                    };
                    const float3 dz_dTw = {s.x, s.y, 1.0};
                    const float dsx_pz = dL_ds.x / p.z;
                    const float dsy_pz = dL_ds.y / p.z;
                    const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
                    const float3 dL_dk = cross_product(l, dL_dp);
                    const float3 dL_dl = cross_product(dL_dp, k);

                    const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
                    const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
                    const float3 dL_dTw = {
                        px * dL_dk.x + py * dL_dl.x + dL_dz * dz_dTw.x,
                        px * dL_dk.y + py * dL_dl.y + dL_dz * dz_dTw.y,
                        px * dL_dk.z + py * dL_dl.z + dL_dz * dz_dTw.z
                    };
                    dL_dTu_local = {dL_dTu.x, dL_dTu.y, dL_dTu.z};
                    dL_dTv_local = {dL_dTv.x, dL_dTv.y, dL_dTv.z};
                    dL_dTw_local = {dL_dTw.x, dL_dTw.y, dL_dTw.z};

                } else {
                    // Update gradient w.r.t center of Gaussian 2D mean position
                    const float dG_ddelx = -vis * FilterInvSquare * d.x;
                    const float dG_ddely = -vis * FilterInvSquare * d.y;
                    v_mean2D_local = {dL_dG * dG_ddelx, dL_dG * dG_ddely};

                }
                dL_dopacity_local = vis * v_alpha;
            }
            warpSum3(dL_drgb_local, warp);
            warpSum(dL_dopacity_local, warp);
            warpSum2(v_mean2D_local, warp);
            warpSum3(dL_dTu_local, warp);
            warpSum3(dL_dTv_local, warp);
            warpSum3(dL_dTw_local, warp);
            if (warp.thread_rank() == 0) {
                // printf("Here!!! \n");
                int32_t g = id_batch[t];
                float* dL_drgb_ptr = (float*)(dL_drgb);
                atomicAdd(dL_drgb_ptr + 3 * g + 0, dL_drgb_local.x);
                atomicAdd(dL_drgb_ptr + 3 * g + 1, dL_drgb_local.y);
                atomicAdd(dL_drgb_ptr + 3 * g + 2, dL_drgb_local.z);
                if (rho3d <= rho2d) {
                    float* dL_dtransMat_ptr = (float*)(dL_dtransMat);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 0, dL_dTu_local.x);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 1, dL_dTu_local.y);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 2, dL_dTu_local.z);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 3, dL_dTv_local.x);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 4, dL_dTv_local.y);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 5, dL_dTv_local.z);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 6, dL_dTw_local.x);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 7, dL_dTw_local.y);
                    atomicAdd(dL_dtransMat_ptr + 9 * g + 8, dL_dTw_local.z);

                    // printf("%.2f, %.2f, %.2f \n", dL_dTw_local.x, dL_dTw_local.y, dL_dTw_local.z);

                    float* v_u_transform_ptr = (float*)(v_u_transforms);
                    atomicAdd(v_u_transform_ptr + 3 * g + 0, dL_dTu_local.x);
                    atomicAdd(v_u_transform_ptr + 3 * g + 1, dL_dTu_local.y);
                    atomicAdd(v_u_transform_ptr + 3 * g + 1, dL_dTu_local.z);

                    float* v_v_transform_ptr = (float*)(v_v_transforms);
                    atomicAdd(v_v_transform_ptr + 3 * g + 0, dL_dTv_local.x);
                    atomicAdd(v_v_transform_ptr + 3 * g + 1, dL_dTv_local.y);
                    atomicAdd(v_v_transform_ptr + 3 * g + 2, dL_dTv_local.z);

                    float* v_w_transform_ptr = (float*)(v_w_transforms);
                    atomicAdd(v_w_transform_ptr + 3 * g + 0, dL_dTw_local.x);
                    atomicAdd(v_w_transform_ptr + 3 * g + 1, dL_dTw_local.y);
                    atomicAdd(v_w_transform_ptr + 3 * g + 2, dL_dTw_local.z);
                } else {
                    float* v_mean2D_ptr = (float*)(v_mean2D);
                    atomicAdd(v_mean2D_ptr + 2 * g + 0, v_mean2D_local.x);
                    atomicAdd(v_mean2D_ptr + 2 * g + 1, v_mean2D_local.y);
                }
                atomicAdd(dL_dopacity + g, dL_dopacity_local);
            }
        }
    }
}


__device__ void build_H(
    const glm::vec3 & p_world,
    const float4 & quat,
    const float2 & scale,
    const float* viewmat,
    const float4 & intrins,
    float tan_fovx,
    float tan_fovy,
    const float* transMat,
    const float3& u_transform,
    const float3& v_transform,
    const float3& w_transform,

    // grad input
    const float* dL_dtransMat,
    const float3& v_u_transform,
    const float3& v_v_transform,
    const float3& v_w_transform,
    // const float* dL_dnormal3D,

    // grad output
    glm::vec3 & v_mean3D,
    glm::vec2 & v_scale,
    glm::vec4 & v_quat
) {

    // camera information
    const glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[1], viewmat[2],
        viewmat[4], viewmat[5], viewmat[6],
        viewmat[8], viewmat[9], viewmat[10]
    ); // viewmat

    const glm::vec3 cam_pos = glm::vec3(viewmat[3], viewmat[7], viewmat[11]); // camera center
    glm::mat4 P = glm::mat4(
        intrins.x, 0.0, 0.0, 0.0,
        0.0, intrins.y, 0.0, 0.0,
        intrins.z, intrins.w, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    );

    glm::mat3 S = scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 RS = R * S;
    glm::vec3 p_view = W * p_world + cam_pos;
    glm::mat3 M = glm::mat3(W * RS[0], W * RS[1], p_view);

    glm::mat4x3 dL_dT = glm::mat4x3(
        dL_dtransMat[0], dL_dtransMat[1], dL_dtransMat[2],
        dL_dtransMat[3], dL_dtransMat[4], dL_dtransMat[5],
        dL_dtransMat[6], dL_dtransMat[7], dL_dtransMat[8],
        0.0, 0.0, 0.0
    );

    // glm::mat4x3 dL_dT = glm::mat4x3(
    //     v_u_transform.x, v_u_transform.y, v_u_transform.z,
    //     v_v_transform.x, v_v_transform.y, v_v_transform.z,
    //     v_w_transform.x, v_w_transform.y, v_w_transform.z,
    //     0.0, 0.0, 0.0
    // );
    // dL_dT = glm::transpose(dL_dT);

    // printf(" \n \
    //         %.5f, %.5f, %.5f \n \
    //         %.5f, %.5f, %.5f \n \
    //         %.5f, %.5f, %.5f \n",
    //         dL_dT[0].x, dL_dT[0].y, dL_dT[0].z,
    //         dL_dT[1].x, dL_dT[1].y, dL_dT[1].z,
    //         dL_dT[2].x, dL_dT[2].y, dL_dT[2].z);

    // printf(" \n \
    //         %.5f, %.5f, %.5f \n \
    //         %.5f, %.5f, %.5f \n \
    //         %.5f, %.5f, %.5f \n",
    //         v_u_transform.x, v_u_transform.y, v_u_transform.z,
    //         v_v_transform.x, v_v_transform.y, v_v_transform.z,
    //         v_w_transform.x, v_w_transform.y, v_w_transform.z);


    glm::mat3x4 dL_dM_aug = glm::transpose(P) * glm::transpose(dL_dT);
    glm::mat3 dL_dM = glm::mat3(
        glm::vec3(dL_dM_aug[0]),
        glm::vec3(dL_dM_aug[1]),
        glm::vec3(dL_dM_aug[2])
    );

    glm::mat3 W_t = glm::transpose(W);
    glm::mat3 dL_dRS= W_t * dL_dM;
    glm::vec3 dL_dRS0 = dL_dRS[0];
    glm::vec3 dL_dRS1 = dL_dRS[1];
    glm::vec3 dL_dpw = dL_dRS[2];
    // glm::vec3 dL_dtn = W_t * glm::vec3(dL_dnormal3D[0], dL_dnormal3D[1], dL_dnormal3D[2]);
    glm::vec3 dL_dtn = W_t * glm::vec3(0.0, 0.0, 0.0);

    
    glm::mat3 dL_dR = glm::mat3(
        dL_dRS0 * glm::vec3(scale.x),
        dL_dRS1 * glm::vec3(scale.y),
        dL_dtn
    );

    v_quat = quat_to_rotmat_vjp(quat, dL_dR);
    v_scale = glm::vec2(
        (float)glm::dot(dL_dRS0, R[0]),
        (float)glm::dot(dL_dRS1, R[1])
    );

    v_mean3D = dL_dpw;
}


__device__ void build_AABB(
    int idx,
    const int * radii,
    const float W, 
    const float H,
    const float * transMats,
    const float3* u_transforms,
    const float3* v_transforms,
    const float3* w_transforms,

    // grad output
    float3 * v_mean2Ds,
    float * dL_dtransMats,
    float3* v_u_transforms,
    float3* v_v_transforms,
    float3* v_w_transforms
) {
    const float* transMat = transMats + 9 * idx;

    const float3 u_transform = u_transforms[idx];
    const float3 v_transform = v_transforms[idx];
    const float3 w_transform = w_transforms[idx];

    const float3 v_mean2D = v_mean2Ds[idx];
    // glm::mat4x3 T = glm::mat4x3(
    //     transMat[0], transMat[1], transMat[2],
    //     transMat[3], transMat[4], transMat[5],
    //     transMat[6], transMat[7], transMat[8],
    //     transMat[6], transMat[7], transMat[8]
    // );

    glm::mat4x3 T = glm::mat4x3(
        u_transform.x, u_transform.y, u_transform.z,
        v_transform.x, v_transform.y, v_transform.z,
        w_transform.x, w_transform.y, w_transform.z,
        w_transform.x, w_transform.y, w_transform.z
    );

    float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
    glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

    glm::vec3 p = glm::vec3(
        glm::dot(f, T[0] * T[3]),
        glm::dot(f, T[1] * T[3]),
        glm::dot(f, T[2] * T[3])
    );

    glm::vec3 dL_dT0 = v_mean2D.x * f * T[3];
    glm::vec3 dL_dT1 = v_mean2D.y * f * T[3];
    glm::vec3 dL_dT3 = v_mean2D.x * f * T[0] + v_mean2D.y * f * T[1];
    glm::vec3 dL_df = (v_mean2D.x * T[0] * T[3]) + (v_mean2D.y * T[1] * T[3]);
    float dL_dd = glm::dot(dL_df, f) * (-1.0 / d);
    glm::vec3 dd_dT3 = glm::vec3(1.0, 1.0, -1.0) * T[3] * 2.0f;
    dL_dT3 += dL_dd * dd_dT3;
    dL_dtransMats[9 * idx + 0] += dL_dT0.x;
    dL_dtransMats[9 * idx + 1] += dL_dT0.y;
    dL_dtransMats[9 * idx + 2] += dL_dT0.z;
    dL_dtransMats[9 * idx + 3] += dL_dT1.x;
    dL_dtransMats[9 * idx + 4] += dL_dT1.y;
    dL_dtransMats[9 * idx + 5] += dL_dT1.z;
    dL_dtransMats[9 * idx + 6] += dL_dT3.x;
    dL_dtransMats[9 * idx + 7] += dL_dT3.y;
    dL_dtransMats[9 * idx + 8] += dL_dT3.z;

    v_u_transforms[idx].x += dL_dT0.x;
    v_u_transforms[idx].y += dL_dT0.y;
    v_u_transforms[idx].z += dL_dT0.z;
    v_v_transforms[idx].x += dL_dT1.x;
    v_v_transforms[idx].y += dL_dT1.y;
    v_v_transforms[idx].z += dL_dT1.z;
    v_w_transforms[idx].x += dL_dT3.x;
    v_w_transforms[idx].y += dL_dT3.y;
    v_w_transforms[idx].z += dL_dT3.z;

    // just use to hack the projected 2D gradient here.
    // float z = transMat[8];
    // v_mean2Ds[idx].x = dL_dtransMats[9 * idx + 2] * z * W; // to ndc
    // v_mean2Ds[idx].y = dL_dtransMats[9 * idx + 5] * z * H; // to ndc
}
