struct CsmData {
    matrix proj_view[4];
    float4 split_distances;
}

float sample_csm(
    Texture2DArray csm_t, 
    SamplerComparisonState comp_shadow_s, 
    float4 shadow_pos_h, 
    int cascade_idx
) {
    float3 proj_coords = shadow_pos_h.xyz / shadow_pos_h.w;

    proj_coords.x = proj_coords.x * 0.5 + 0.5;
    proj_coords.y = -proj_coords.y * 0.5 + 0.5;

    float3 tex_coord;
    tex_coord.xy = proj_coords.xy;
    tex_coord.z = cascade_idx;

    return csm_t.SampleCmpLevelZero(comp_shadow_s, tex_coord, proj_coords.z).r;
}

float sample_csm_with_pcf(
    Texture2DArray csm_t, 
    SamplerComparisonState comp_shadow_s, 
    float4 shadow_pos_h, 
    int cascade_idx
) {
    float3 proj_coords = shadow_pos_h.xyz / shadow_pos_h.w;

    proj_coords.x = proj_coords.x * 0.5 + 0.5;
    proj_coords.y = -proj_coords.y * 0.5 + 0.5;

    float3 tex_coord;
    tex_coord.xy = proj_coords.xy;
    tex_coord.z = cascade_idx;

    float dx = ddx(proj_coords.x);

    float percent_lit = 0.0f;
    const float3 offsets[9] =
    {
        float3(-dx,  -dx, 0.0f), float3(0.0f,  -dx, 0.0f), float3(dx,  -dx, 0.0f),
        float3(-dx, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(dx, 0.0f, 0.0f),
        float3(-dx,  +dx, 0.0f), float3(0.0f,  +dx, 0.0f), float3(dx,  +dx, 0.0f)
    };

    [unroll]
    for(int i = 0; i < 9; ++i)
    {
        percent_lit += csm_t.SampleCmpLevelZero(comp_shadow_s, tex_coord+ offsets[i], proj_coords.z).r;
    }
    
    return percentLit / 9.0f;
}