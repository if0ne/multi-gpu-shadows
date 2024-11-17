struct PixelInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD;
    float3 normal : NORMAL;
};

float4 Main(PixelInput input) : SV_TARGET
{
    return float4(1.0, 1.0, 1.0, 1.0);
}