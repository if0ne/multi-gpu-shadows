#include "Common.hlsl"
#include "ShadowCommon.hlsl"
#include "FullscreenVS.hlsl"

cbuffer GlobalBuffer : register(b0) {
    Global g_data;
}

cbuffer CsmBuffer : register(b1) {
    CsmData csm_data;
}

Texture2D depth_t : register(t2);
Texture2DArray csm_t : register(t3);

SamplerComparisonState comp_shadow_s : register(s0);

float1 Main(FullscreenVertex input) : SV_TARGET {
    int2 tex_coord = input.pos.xy;
    float depth = depth_t.Load(int3(texCoord, 0)).r;

    float4 world_pos = screen_to_world(float4(tex_coord, depth, 1.0f), g_data.screen_dim, g_data.inv_proj_view);

    float fragment_dist = length(g_data.eye_pos - world_pos.xyz);
    uint cascade_idx = 0;

     for (uint i = 0; i < 4 - 1; ++i)
    {
        if (fragment_dist > csm_data.split_distances[i])
        {
            cascade_idx = i + 1;
        }
    }

    float shadow_factor = sample_csm(csm_t, comp_shadow_s, mul(csm_data.proj_view[cascade_idx], world_pos), cascade_idx);

    return shadow_factor;
}
