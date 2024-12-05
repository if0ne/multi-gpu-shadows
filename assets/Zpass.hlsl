#include "Common.hlsl"
#include "Pbr.hlsl"

cbuffer GlobalBuffer : register(b0) {
    Globals g_data;
}

struct VertexInput {
    float3 pos : POSITION;
};

struct PixelInput {
    float4 pos : SV_POSITION;
};

PixelInput Main(VertexInput input) {
    PixelInput output = (PixelInput) 0;
	
    float4 world_pos = float4(input.pos, 1.0f);
    output.pos = mul(g_data.proj_view, world_pos);
	
    return output;
}
