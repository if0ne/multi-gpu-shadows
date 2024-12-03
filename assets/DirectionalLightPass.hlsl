#include "Common.hlsl"
#include "Light.hlsl"
#include "Pbr.hlsl"
#include "FullscreenVS.hlsl"

cbuffer GlobalBuffer : register(b0) {
    Global g_data;
}

cbuffer DirectionalLightBuffer : register(b1)
{
    DirectionalLight dir_light;
};

cbuffer AmbientLightBuffer : register(b2)
{
    AmbientLight ambient_light;
}

Texture2D diffuse_t : register(t3);
Texture2D normal_t : register(t4);
Texture2D material_t : register(t5);
Texture2D shadow_mask_t : register(t6);

float3 compute_dir_light(float4 diffuse, float3 normal, float3 to_eye, float4 material) {
    float3 lightVec = -dir_light.direction;

    float ndotl = max(dot(lightVec, normal), 0.0f);
    float3 light_strength = dir_light.strength * ndotl;

    MaterialEntry mat;
    mat.fresnel_r0 = material.x;
    mat.roughness = material.y;

    return blinn_phong(diffuse, light_strength, lightVec, normal, to_eye);
}

float4 Main(FullscreenVertex input) : SV_Target {
    int2 tex_coord = input.pos.xy;
   
    float4 diffuse = diffuse_t.Load(int3(tex_coord, 0));
    float3 normal = unpack_normal_from_texture(normal_t.Load(int3(tex_coord, 0)));
    float4 material = material_t.Load(int3(tex_coord, 0));
    float shadow_factor = shadow_mask_t.Load(int3(tex_coord, 0)).r;

    float depth = material.w;
    float4 world_pos = screen_to_world(float4(texCoord, depth, 1.0f), g_data.screen_dim, g_data.inv_proj_view);
    
    float3 to_eye = normalize(g_data.eye_pos - world_pos.xyz);
    float4 ambient = ambient_light.color * diffuse;

    float4 dir_light_color = float4(shadow_factor * compute_dir_light(diffuse, normal, to_eye, material), 1.0);
	float4 lit_color = ambient + dir_light_color;

	lit_color.a = diffuse.a;
    return lit_color;
}
