cbuffer Mat : register(b1)
{
    float4 diffuse;
}

struct PixelInput
{
    float4 position : SV_POSITION;
    float3 normal: NORMAL;
};

float4 Main(PixelInput input) : SV_TARGET
{
    return diffuse;
}