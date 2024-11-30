Texture2D gDiffuseMap : register(t2);
SamplerState gsamLinearClamp : register(s0);

cbuffer Mat : register(b1)
{
    float4 diffuse;
}

struct PixelInput
{
    float4 position : SV_POSITION;
    float3 normal: NORMAL;
    float2 uv : TEXCOORD;
};

float4 Main(PixelInput input) : SV_TARGET
{
    float4 color = gDiffuseMap.Sample(gsamLinearClamp, input.uv) * diffuse;
    return diffuse;
}